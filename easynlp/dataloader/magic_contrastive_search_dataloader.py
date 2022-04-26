from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class MAGICGPT2Dataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.sos_token = '<-start_of_text->'
        self.pad_token = '<-pad->'
        self.vocab.add_tokens([self.sos_token, self.pad_token])
        self.sos, self.pad = self.vocab.convert_tokens_to_ids([self.sos_token, self.pad_token])
        print ('sos token is {}, sos token id is {}'.format(self.sos_token, self.sos))
        print ('pad token is {}, pad token id is {}'.format(self.pad_token, self.pad))
        self.eos_token, self.eos = self.vocab.bos_token, self.vocab.bos_token_id
        print ('eos token is {}, eos token id is {}'.format(self.eos_token, self.eos))

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        line = json.loads(line)['captions']
        try:
            tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
            tokens = [self.sos] + tokens + [self.eos]
            return tokens
        except:
            return None

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in batch if i]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            ids, ods, ids_mask = ids[:, :-1], ids[:, 1:], ids_mask[:, :-1]
            ids, ods, ids_mask = to_cuda(ids, ods, ids_mask)
            return {
                'ids': ids, 
                'ods': ods,
                'ids_mask': ids_mask, 
            }
        else:
            max_length = max([len(i) for i in batch])
            ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in batch])
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'ids_label': ids,
                'pos_ids': pos_ids, 
            }
