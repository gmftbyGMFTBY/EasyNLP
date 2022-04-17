from header import *
from itertools import islice
from .utils import *
from .util_func import *
from .randomaccess import *


class FilterInferenceDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        self.r_data_root_path = args['r_data_root_path']
        self.w_data_root_path = args['w_data_root_path']
        self.file_lists = [
            (os.path.join(self.r_data_root_path, file), os.path.join(self.w_data_root_path, file) )
            for file in os.listdir(self.r_data_root_path)
        ]
        self.file_lists = sorted(self.file_lists, key=lambda x:x[0])
        subjob_num = len(self.file_lists) // args['nums']
        jobs = []
        for i in range(0, len(self.file_lists), subjob_num):
            jobs.append(self.file_lists[i:i+subjob_num])
        if len(jobs) > args['nums']:
            jobs[-2].extend(jobs[-1])
            jobs = jobs[:-1]
        assert len(jobs) == args['nums']

        # count lines
        self.job = jobs[args['local_rank']]
        self.size = 0
        for path, _ in tqdm(jobs):
            self.size += iter_count(path)
        print(f'[!] worker {self.args["local_rank"]} gets {self.size} samples to clean')
        # 
        self.buff_size = args['buff_size']
        self.cache = []
        self.file_index = 0
        self.init_file()

    def init_file(self):
        # close the previous file
        if self.file_index > 0:
            self.current_r_file.close()
            self.current_w_file.close()
        # open the new file
        if self.file_index >= len(self.job):
            raise StopIteration
        self.current_r_file, self.current_w_file = self.job[self.file_index]
        self.current_r_file = open(self.current_r_file)
        self.current_w_file = open(self.current_w_file, 'w')
        self.file_index += 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):

        # read one line from the cache
        if len(self.cache) > 0:
            line = self.cache.pop()
        else:
            self.cache = list(islice(self.current_r_file, self.buff_size))
            if len(self.cache) == 0:
                self.init_file()
                self.cache = list(islice(self.current_r_file, self.buff_size))
            line = self.cache.pop()

        item = json.loads(line)
        data = item['data']
        try:
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
        except:
            return None, None, None
        response = items[-1]
        context = []
        for u in items[:-1]:
            context.extend(u + [self.sep])
        context.pop()
        truncate_pair(context, response, self.args['max_len'])
        ids = [self.cls] + context + [self.sep] + response + [self.sep]
        tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
        return ids, tids, item

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, raw = [], [], []
        for a, b, c in batch:
            if a and b:
                ids.append(torch.LongTensor(a))
                tids.append(torch.LongTensor(b))
                raw.append(c)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
        return {
            'ids': ids, 
            'tids': tids,
            'mask': ids_mask, 
            'raw': raw,
        }


class FilterDRInferenceDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        self.r_data_root_path = args['r_data_root_path']
        self.w_data_root_path = args['w_data_root_path']
        self.file_lists = [
            (os.path.join(self.r_data_root_path, file), os.path.join(self.w_data_root_path, file) )
            for file in os.listdir(self.r_data_root_path)
        ]
        self.file_lists = sorted(self.file_lists, key=lambda x:x[0])
        
        # count lines
        self.size = 0
        for path, _ in tqdm(self.file_lists):
            self.size += iter_count(path)
        print(f'[!] worker {self.args["local_rank"]} gets {self.size} samples to clean')

        subjob_num = len(self.file_lists) // args['nums']
        jobs = []
        for i in range(0, len(self.file_lists), subjob_num):
            jobs.append(self.file_lists[i:i+subjob_num])
        if len(jobs) > args['nums']:
            jobs[-2].extend(jobs[-1])
            jobs = jobs[:-1]
        assert len(jobs) == args['nums']

        self.job = jobs[args['local_rank']]
        self.buff_size = args['buff_size']
        self.cache = []
        self.file_index = 0
        self.init_file()

    def init_file(self):
        # close the previous file
        if self.file_index > 0:
            self.current_r_file.close()
            self.current_w_file.close()
        # open the new file
        if self.file_index >= len(self.job):
            raise StopIteration
        self.current_r_file, self.current_w_file = self.job[self.file_index]
        self.current_r_file = open(self.current_r_file)
        self.current_w_file = open(self.current_w_file, 'w')
        self.file_index += 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):

        # read one line from the cache
        if len(self.cache) > 0:
            line = self.cache.pop()
        else:
            self.cache = list(islice(self.current_r_file, self.buff_size))
            if len(self.cache) == 0:
                self.init_file()
                self.cache = list(islice(self.current_r_file, self.buff_size))
            line = self.cache.pop()

        item = json.loads(line)
        data = item['data']
        try:
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
        except:
            return None, None, None
        response = items[-1]
        context = []
        for u in items[:-1]:
            context.extend(u + [self.sep])
        context.pop()
        c, r = context[-self.args['max_len']:], response[:self.args['res_max_len']]
        ids = [self.cls] + c + [self.sep]
        rids = [self.cls] + r + [self.sep]
        return ids, rids, item

    def save(self):
        pass
        
    def collate(self, batch):
        ids, rids, raw = [], [], []
        for a, b, c in batch:
            if a and b:
                ids.append(torch.LongTensor(a))
                rids.append(torch.LongTensor(b))
                raw.append(c)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        rids_mask = generate_mask(rids, pad_token_idx=self.pad)
        ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
        return {
            'ids': ids, 
            'rids': rids,
            'ids_mask': ids_mask, 
            'rids_mask': rids_mask, 
            'raw': raw,
        }
