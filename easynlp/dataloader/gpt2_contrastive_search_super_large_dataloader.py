from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class GPT2ForContrastiveForBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.size = 10000000
        path = f'/apdcephfs/share_733425/johntianlan/chinese_high_quality_300g_split/train_{args["global_rank"]}.txt'
        # path = f'/apdcephfs/share_916081/johntianlan/chinese_high_quality_300g_test/train_{args["global_rank"]}.txt'
        self.path = path
        self.reader = open(path)
        self.buffersize = 40960000
        self.buffer = StringIO('')
        print(f'[!] dataset size: {self.size} for file {self.path}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        cache = ''
        while True:
            line = self.buffer.readline()
            if line and line[-1] == '\n':
                line = cache + line
                break
            else:
                strings = self.reader.read(self.buffersize)
                if not strings:
                    # reopen the file
                    self.reader = open(self.path)
                self.buffer = StringIO(strings)
                cache = cache + line
        line = eval(line.strip())['content']
        line = line.replace('\n', ' ' + self.vocab.sep_token + ' ')
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

class GPT2ForContrastiveForBigArxivDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.bos_token_id
        self.size = 10000000
        path = f'/apdcephfs/share_916081/johntianlan/arxiv_data/rank_0{args["global_rank"]}'
        self.path = path
        self.reader = open(path, encoding='utf-8')
        self.buffersize = 409600000
        self.buffer = StringIO('')
        print(f'[!] dataset size: {self.size} for file {self.path}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        cache = ''
        try_num = 0
        while True:
            line = self.buffer.readline().strip()
            if line:
                cache = cache + line
                try:
                    data = eval(cache)
                    break
                except:
                    try_num += 1
                    if try_num > 5:
                        cache = ''
                    continue
            else:
                strings = self.reader.read(self.buffersize)
                if not strings:
                    self.reader = open(self.path)
                    strings = self.reader.read(self.buffersize)
                self.buffer = StringIO(strings)
        line = data['contents']
        line = random.choice(line.split('\n'))
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
