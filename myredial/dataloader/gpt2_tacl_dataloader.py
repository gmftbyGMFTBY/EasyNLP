from header import *
from .utils import *
from .util_func import *


class GPT2TaCLDataset(Dataset):

    '''gpt2 and roberta has must share the same vocabulary'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

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
        while True:
            line = self.reader.get_line(i)
            sentences = json.loads(line.strip())['q']
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentence) > 0:
                break
            i = random.choice(range(self.size))

        sentences = [''.join(sentence.split()) for sentence in sentences]
        tokens = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
        tokens = list(chain(*tokens))
        # sample the max_length sequence from it
        if len(tokens) > self.args['max_len']:
            sample_range = list(range(0, len(tokens) - self.args['max_len']))
            head = random.choice(sample_range)
            tail = head += self.args['max_len']
            tokens = tokens[head:tail]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            batch = [torch.LongTensor(i) for i in batch]
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, ids_mask = to_cuda(ids, ids_mask)
        else:
            # batch size must be 1 
            # sample the prefix
            assert len(batch) == 1
            tokens = batch[0][:-self.args['prefix_len']]
            ids = torch.LongTensor(tokens).unsqueeze(0)    # [1, S]
            ids_mask = torch.ones_like(ids)
            ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }
