from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class GPT2ForContrastiveForBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        if self.args['mode'] == 'train':
            self.path = f'/apdcephfs/share_733425/johntianlan/chinese_high_quality_300g_split/train_{args["global_rank"]}.txt'
            # self.path = f'/apdcephfs/share_916081/johntianlan/chinese_high_quality_300g_test/train_{args["global_rank"]}.txt'
            self.current_file_handler = open(self.path, 'r')

            self.size = 10000000
            self.cache = []
            self.buffer_size = 4096000
        else:
            path = f'/apdcephfs/share_733425/johntianlan/chinese_high_quality_300g_split/test.txt'
            # path = f'/apdcephfs/share_916081/johntianlan/chinese_high_quality_300g_test/test.txt'
            with open(path) as f:
                dataset = []
                for line in f.readlines():
                    dataset.append(eval(line.strip()))
            self.data = dataset
            self.size = len(dataset)
            print(f'[!] load {len(dataset)} test samples')
                
    def __len__(self):
        return self.size

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_handler = open(self.path, 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        # shuffle
        random.shuffle(self.cache)

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) == 0:
                self.load_one_chunk()
            line = eval(self.cache.pop())['content']
            line = line.replace('\n', ' ')
            if self.args['mode'] in ['train']:
                tokens = self.vocab.encode(line, add_special_tokens=False)
                if len(tokens) > self.args['max_len']:
                    sample_range = range(0, len(tokens) - self.args['max_len'])
                    index = random.choice(sample_range)
                    tokens = tokens[index:index+self.args['max_len']]
            else:
                tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['prefix_len']]
            return tokens
        else:
            line = self.data[i]['content']
            tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
            return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [torch.LongTensor(i) for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ods, ids_mask = ids[:, :-1], ids[:, 1:], ids_mask[:, :-1]
        ids, ods, ids_mask = to_cuda(ids, ods, ids_mask)
        return {'ids': ids, 'ods': ods, 'ids_mask': ids_mask}


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


class GPT2ForContrastiveCommonCrawlForBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.bos_token_id

        if self.args['mode'] == 'train':
            self.size = 10000000
            self.file_lists = [f'{self.args["data_root_path"]}/train_{i}.txt' for i in range(16)]

            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)

            self.current_file_index = 0
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = []
            self.buffer_size = args['buffer_size']
        else:
            self.file = f'{self.args["data_root_path"]}/test.txt'
            with open(self.file) as f:
                dataset = []
                for line in f.readlines():
                    if line.strip():
                        dataset.append(line.strip())
            self.data = dataset
            self.size = len(self.data)
            print(f'[!] collect {len(dataset)} test samples')

    def __len__(self):
        return self.size

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == 7 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            tokens = []
            # read to the max_length or the document ending accure
            while len(tokens) < self.args['max_len']:
                if len(self.cache) == 0:
                    self.load_one_chunk()
                line = self.cache[0].strip()
                if line.strip():
                    t = self.vocab.encode(line, add_special_tokens=False)
                    if len(tokens) + len(t) > self.args['max_len']:
                        delta = self.args['max_len'] - len(tokens)
                        tokens += t[:delta]
                        t = t[delta:]
                        raw_text = self.vocab.decode(t)
                        self.cache[0] = raw_text
                        break
                    else:
                        tokens.extend(t)
                        self.cache.pop(0)
                else:
                    self.cache.pop(0)
                    if len(tokens) > 0:
                        break
            return tokens
        else:
            line = self.data[i]
            tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
            return tokens

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [torch.LongTensor(i) for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ods, ids_mask = ids[:, :-1], ids[:, 1:], ids_mask[:, :-1]
        ids, ods, ids_mask = to_cuda(ids, ods, ids_mask)
        return {'ids': ids, 'ods': ods, 'ids_mask': ids_mask}


