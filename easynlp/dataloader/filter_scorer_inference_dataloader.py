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

        self.size = 0
        self.valid_file_lists = []
        for r_path, w_path in tqdm(self.file_lists):
            if r_path.endswith('txt') is False:
                continue
            valid, size = self.check_valid(r_path, w_path)
            if valid is False:
                self.size += size
                self.valid_file_lists.append((r_path, w_path))
        self.file_lists = self.valid_file_lists
        print(f'[!] gets {self.size} valid samples to clean')

        subjob_num = len(self.file_lists) // args['nums']
        jobs = []
        for i in range(0, len(self.file_lists), subjob_num):
            jobs.append(self.file_lists[i:i+subjob_num])
        if len(jobs) > args['nums']:
            ext_job = jobs[-1]
            jobs = jobs[:-1]
            for i in range(len(ext_job)):
                jobs[i].append(ext_job[i])
        assert len(jobs) == args['nums']

        # count lines
        self.job = jobs[args['local_rank']]
        self.buff_size = args['buff_size']
        self.cache = []
        self.file_index = 0
        self.init_file()

    def check_valid(self, r_path, w_path):
        r_counter = iter_count(r_path)
        w_counter = iter_count(w_path)
        if r_counter == w_counter:
            return True, r_counter
        else:
            return False, r_counter

    def init_file(self):
        if self.file_index >= len(self.job):
            return
        # open the new file
        self.current_r_file, self.current_w_file = self.job[self.file_index]
        self.current_r_file = open(self.current_r_file)
        self.current_w_file = open(self.current_w_file, 'w')
        self.file_index += 1

    def __len__(self):
        return self.size * 2

    def __getitem__(self, i):

        # read one line from the cache
        try:
            if len(self.cache) > 0:
                line = self.cache.pop()
            else:
                self.cache = list(islice(self.current_r_file, self.buff_size))
                if len(self.cache) == 0:
                    self.init_file()
                    self.cache = list(islice(self.current_r_file, self.buff_size))
                line = self.cache.pop()
        except:
            return None, None, None, None
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
        return ids, tids, item, self.current_w_file

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, raw, writers = [], [], [], []
        for a, b, c, d in batch:
            if a and b:
                ids.append(torch.LongTensor(a))
                tids.append(torch.LongTensor(b))
                raw.append(c)
                writers.append(d)
        if len(ids) == 0:
            return {
                'ids': None,
                'rids': None,
                'ids_mask': None,
                'rids_mask': None,
                'raws': None,
                'writers': None
            }
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
        return {
            'ids': ids, 
            'tids': tids,
            'mask': ids_mask, 
            'raw': raw,
            'writers': writers
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
        print(f'[!] total samples for inference: {self.size}')

        subjob_num = len(self.file_lists) // args['nums']
        jobs = []
        for i in range(0, len(self.file_lists), subjob_num):
            jobs.append(self.file_lists[i:i+subjob_num])
        if len(jobs) > args['nums']:
            ext_job = jobs[-1]
            jobs = jobs[:-1]
            for i in range(len(ext_job)):
                jobs[i].append(ext_job[i])
        assert len(jobs) == args['nums']

        self.job = jobs[args['local_rank']]
        self._size = 0
        for path, _ in self.job:
            self._size += iter_count(path)
        print(f'[!] worker {self.args["local_rank"]} gets {self._size} samples ({len(self.job)} files) to clean')

        self.buff_size = args['buff_size']
        self.cache = []
        self.file_index = 0
        self.init_file()
        self.error_count = 0

    def init_file(self):
        # open the new file
        if self.file_index >= len(self.job):
            return
        self.current_r_file, self.current_w_file = self.job[self.file_index]
        self.current_r_file = open(self.current_r_file)
        self.current_w_file = open(self.current_w_file, 'w')
        self.file_index += 1

    def __len__(self):
        return self.size * 2

    def __getitem__(self, i):

        # read one line from the cache
        try:
            if len(self.cache) > 0:
                line = self.cache.pop()
            else:
                self.cache = list(islice(self.current_r_file, self.buff_size))
                if len(self.cache) == 0:
                    self.init_file()
                    self.cache = list(islice(self.current_r_file, self.buff_size))
                line = self.cache.pop()
        except:
            return None, None, None, None
        fw = self.current_w_file
        item = json.loads(line)
        data = item['data']
        try:
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
        except:
            self.error_count += 1
            return None, None, None, None
        response = items[-1]
        context = []
        for u in items[:-1]:
            context.extend(u + [self.sep])
        context.pop()
        c, r = context[-self.args['max_len']:], response[:self.args['res_max_len']]
        ids = [self.cls] + c + [self.sep]
        rids = [self.cls] + r + [self.sep]
        return ids, rids, item, self.current_w_file

    def save(self):
        pass
        
    def collate(self, batch):
        ids, rids, raw, writers = [], [], [], []
        for a, b, c, d in batch:
            if a and b:
                ids.append(torch.LongTensor(a))
                rids.append(torch.LongTensor(b))
                raw.append(c)
                writers.append(d)
        if len(ids) == 0:
            return {
                'ids': None,
                'rids': None,
                'ids_mask': None, 
                'rids_mask': None, 
                'raw': None,
                'writers': None 
            }
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
            'writers': writers
        }


class FilterInferenceCombinationDataset(Dataset):

    '''for bert-ft and dr-bert models'''

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
            for file in os.listdir(self.r_data_root_path) if file.endswith('txt')
        ]
        self.file_lists = sorted(self.file_lists, key=lambda x:x[0])

        self.size = 0
        self.valid_file_lists = []
        for r_path, w_path in tqdm(self.file_lists):
            valid, size = self.check_valid(r_path, w_path)
            if valid is False:
                self.size += size
                self.valid_file_lists.append((r_path, w_path))
        self.file_lists = self.valid_file_lists
        print(f'[!] {len(self.file_lists)} file founds, gets {self.size} invalid samples to clean')

        subjob_num = max(1, len(self.file_lists) // args['nums'])
        jobs = []
        for i in range(0, len(self.file_lists), subjob_num):
            jobs.append(self.file_lists[i:i+subjob_num])
        if len(jobs) > args['nums']:
            delta = len(jobs) - args['nums']
            ext_job = jobs[-delta:]
            jobs = jobs[:-delta]
            counter = 0
            for ext_job_ in ext_job:
                for i in range(len(ext_job_)):
                    jobs[counter].append(ext_job_[i])
                    counter += 1
        assert len(jobs) <= args['nums'], f'{len(jobs)} jobs and {args["nums"]} workers'

        # count lines
        try:
            self.job = jobs[args['local_rank']]
        except:
            return 
        self.buff_size = args['buff_size']
        self.cache = []
        self.file_index = 0
        self.init_file()

    def init_file(self):
        if self.file_index >= len(self.job):
            return
        # open the new file
        self.current_r_file, self.current_w_file = self.job[self.file_index]
        self.current_r_file = open(self.current_r_file)
        self.current_w_file = open(self.current_w_file, 'w')
        self.file_index += 1

    def __len__(self):
        return self.size * 2

    def __getitem__(self, i):
        try:
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
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            # for bert-ft
            response = items[-1]
            context = []
            for u in items[:-1]:
                context.extend(u + [self.sep])
            context.pop()
            truncate_pair(context, response, self.args['max_len'])
            ids = [self.cls] + context + [self.sep] + response + [self.sep]
            tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
            # for dr-bert
            response = items[-1]
            context = []
            for u in items[:-1]:
                context.extend(u + [self.sep])
            context.pop()
            c, r = context[-self.args['ctx_max_len']:], response[:self.args['res_max_len']]
            ids_ = [self.cls] + c + [self.sep]
            rids_ = [self.cls] + r + [self.sep]
        except:
            return None, None, None, None, None, None
        return ids, tids, ids_, rids_, item, self.current_w_file

    def save(self):
        pass
    
    def check_valid(self, r_path, w_path):
        r_counter = iter_count(r_path)
        if os.path.exists(w_path) is False:
            return False, r_counter
        try:
            w_counter = iter_count(w_path)
        except:
            return False, r_counter
        if r_counter == w_counter:
            return True, r_counter
        else:
            return False, r_counter
        
    def collate(self, batch):
        ids, tids, ids_, rids_, raw, writers = [], [], [], [], [], []
        for a, b, c, d, e, f in batch:
            if a and b:
                ids.append(torch.LongTensor(a))
                tids.append(torch.LongTensor(b))
                ids_.append(torch.LongTensor(c))
                rids_.append(torch.LongTensor(d))
                raw.append(e)
                writers.append(f)
        if len(ids) == 0:
            return {
                'ids': None,
                'tids': None,
                'ids_': None,
                'rids_': None,
                'ids_mask': None,
                'ids_mask_': None,
                'rids_mask_': None,
                'raws': None,
                'writers': None
            }
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids_ = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
        rids_ = pad_sequence(rids_, batch_first=True, padding_value=self.pad)
        ids_mask_, rids_mask_ = generate_mask(ids_), generate_mask(rids_)
        ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
        ids_, rids_, ids_mask_, rids_mask_ = to_cuda(ids_, rids_, ids_mask_, rids_mask_)
        return {
            'ids': ids, 
            'tids': tids,
            'ids_': ids_,
            'rids_': rids_,
            'ids_mask': ids_mask,
            'ids_mask_': ids_mask_, 
            'rids_mask_': rids_mask_,
            'raw': raw,
            'writers': writers
        }
