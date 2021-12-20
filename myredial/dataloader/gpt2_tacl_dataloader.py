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
            if len(sentences) > 0:
                break
            i = random.choice(range(self.size))

        sentences = [''.join(sentence.split()) for sentence in sentences]
        tokens = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
        tokens = list(chain(*tokens))
        # sample the max_length sequence from it
        if len(tokens) > self.args['max_len']:
            sample_range = list(range(0, len(tokens) - self.args['max_len']))
            head = random.choice(sample_range)
            tail = head + self.args['max_len']
            tokens = tokens[head:tail]
        return tokens, [self.cls] + tokens + [self.sep]

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i[0]) for i in batch]
            bert_ids = [torch.LongTensor(i[1]) for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            bert_ids = pad_sequence(bert_ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            bert_ids_mask = generate_mask(bert_ids)
            ids, ids_mask, bert_ids, bert_ids_mask = to_cuda(ids, ids_mask, bert_ids, bert_ids_mask)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'bert_ids': bert_ids,
                'bert_ids_mask': bert_ids_mask,
            }
        else:
            # left pad
            batch = [i[0] for i in batch]
            max_length = max([len(i) for i in batch])
            ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in batch])
            ids_mask = generate_mask(ids)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'ids_label': ids,
                'pos_ids': pos_ids, 
            }

            
class GPT2TaCLV2Dataset(Dataset):

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
            if len(sentences) > 0:
                break
            i = random.choice(range(self.size))

        sentences = [''.join(sentence.split()) for sentence in sentences]
        tokens = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
        tokens = list(chain(*tokens))
        # sample the max_length sequence from it
        if len(tokens) > self.args['max_len']:
            sample_range = list(range(0, len(tokens) - self.args['max_len']))
            head = random.choice(sample_range)
            tail = head + self.args['max_len']
            tokens = tokens[head:tail]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, da_ids = [], []
            for tokens in batch:
                ids.append(torch.LongTensor(tokens))
                effective_tokens = tokens[-self.args['max_da_len']:]
                if len(effective_tokens) < len(tokens):
                    da_tokens = [self.pad] * (len(tokens) - len(effective_tokens)) + effective_tokens
                else:
                    da_tokens = effective_tokens
                assert len(da_tokens) == len(tokens)
                da_ids.append(torch.LongTensor(da_tokens))
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            da_ids = pad_sequence(da_ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            da_ids_mask = generate_mask(da_ids)
            da_pos_ids = (da_ids_mask.long().cumsum(-1) - 1).masked_fill(da_ids_mask == self.pad, 0)
            ids, ids_mask, da_ids, da_ids_mask, da_pos_ids = to_cuda(ids, ids_mask, da_ids, da_ids_mask, da_pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'da_ids': da_ids, 
                'da_ids_mask': da_ids_mask, 
                'da_pos_ids': da_pos_ids, 
            }
        else:
            # left pad
            max_length = max([len(i) for i in batch])
            ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in batch])
            ids_mask = generate_mask(ids)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'ids_label': ids,
                'pos_ids': pos_ids, 
            }

            
class GPT2TaCLV3Dataset(Dataset):

    '''based on the GPT2TaCLDataset, but also include the left pad gpt2 inputs for sequecen level unlikelyhood traininig'''
    
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
            if len(sentences) > 0:
                break
            i = random.choice(range(self.size))

        sentences = [''.join(sentence.split()) for sentence in sentences]
        tokens = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
        tokens = list(chain(*tokens))
        # sample the max_length sequence from it
        if len(tokens) > self.args['max_len']:
            sample_range = list(range(0, len(tokens) - self.args['max_len']))
            head = random.choice(sample_range)
            tail = head + self.args['max_len']
            tokens = tokens[head:tail]
        return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            max_length = max([len(i) for i in batch])
            ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in batch])
            ids_mask = torch.stack([torch.LongTensor([0] * (max_length - len(i)) + [1] * len(i)) for i in batch])
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'pos_ids': pos_ids,
            }
        else:
            # left pad
            batch = [i[0] for i in batch]
            max_length = max([len(i) for i in batch])
            ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in batch])
            ids_mask = generate_mask(ids)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'ids_label': ids,
                'pos_ids': pos_ids, 
            }
