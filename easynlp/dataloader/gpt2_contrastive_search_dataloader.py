from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class GPT2WikiTextDataset(Dataset):

    '''wikitext-103 dataset'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.eos_token_id

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
        tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            ids, ids_mask = to_cuda(ids, ids_mask)
            return {
                'ids': ids, 
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


class GPT2WikiTextV2Dataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.eos_token_id

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
        tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
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

class GPT2ForContrastiveDataset(Dataset):

    '''wikitext-103 dataset'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        if args['dataset'] in ['writer-rank', 'chinese_pretrain']:
            self.pad = self.vocab.pad_token_id
        else:
            self.pad = self.vocab.bos_token_id

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
        # for chinese
        if self.args['dataset'] in ['writer-rank', 'chinese_pretrain']:
            while True:
                line = self.reader.get_line(i).strip()
                try:
                    sentences = json.loads(line.strip())['q']
                except:
                    i = random.choice(range(self.size))
                    continue
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) > 0:
                    break
                i = random.choice(range(self.size))
            line = ''.join([''.join(s.strip().split()) for s in sentences])
        else:
            raise Exception()

        if self.args['mode'] in ['train']:
            tokens = self.vocab.encode(line, add_special_tokens=False)
            if len(tokens) > self.args['max_len']:
                sample_range = range(0, len(tokens) - self.args['max_len'])
                index = random.choice(sample_range)
                tokens = tokens[index:index+self.args['max_len']]
        else:
            tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['prefix_len']]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            ids, ods, ids_mask = ids[:, :-1], ids[:, 1:], ids_mask[:, :-1]
            ids, ods, ids_mask = to_cuda(ids, ods, ids_mask)
            return {'ids': ids, 'ods': ods, 'ids_mask': ids_mask}
        else:
            max_length = max([len(i) for i in batch])
            ids = torch.LongTensor([[self.pad] * (max_length - len(i)) + i for i in batch])
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            ids, ods, ids_mask = ids[:, :-1], ids[:, 1:], ids_mask[:, :-1]
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ods, ids_mask, pos_ids = to_cuda(ids, ods, ids_mask, pos_ids)
            return {'ids': ids, 'ids_label': ods, 'ids_mask': ids_mask, 'pos_ids': pos_ids}
