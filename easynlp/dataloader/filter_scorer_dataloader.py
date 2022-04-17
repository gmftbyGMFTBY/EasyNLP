from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class FilterDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id

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
        item = json.loads(line)
        if self.args['mode'] == 'train':
            label = int(item['score'])
            if label not in [0, 1]:
                return None, None, None
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
            return ids, tids, label
        else:
            data = item['data'][:-1]
            candidates = [item['data'][-1]] + item['neg_sentences']
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            context = []
            for u in items:
                context.extend(u + [self.sep])
            context.pop() 
            items = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']

            assert len(items) == 10
            ids, tids, label = [], [], []
            for idx, utter in enumerate(items):
                if idx == 0:
                    label.append(1)
                else:
                    label.append(0)
                c, r = deepcopy(context), deepcopy(utter)
                truncate_pair(c, r, self.args['max_len'])
                ids_ = [self.cls] + c + [self.sep] + r + [self.sep]
                tids_ = [0] * (len(c) + 2) + [1] * (len(r) + 1)
                ids.append(ids_)
                tids.append(tids_)
            return ids, tids, label

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [], [], []
            for a, b, c in batch:
                if a and b:
                    ids.append(torch.LongTensor(a))
                    tids.append(torch.LongTensor(b))
                    label.append(c)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, tids, ids_mask, label = to_cuda(ids, tids, ids_mask, label)
            return {
                'ids': ids, 
                'tids': tids,
                'mask': ids_mask, 
                'label': label
            }
        else:
            assert len(batch) == 1
            ids, tids, label = batch[0]
            ids = pad_sequence([torch.LongTensor(i) for i in ids], batch_first=True, padding_value=self.pad)
            tids = pad_sequence([torch.LongTensor(i) for i in tids], batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, tids, ids_mask, label = to_cuda(ids, tids, ids_mask, label)
            return {
                'ids': ids,
                'tids': tids,
                'mask': ids_mask,
                'label': label,
            }



class FilterDRDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id

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
        item = json.loads(line)
        if self.args['mode'] == 'train':
            data = item['data']
            try:
                items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            except:
                return None, None
            response = items[-1]
            context = []
            for u in items[:-1]:
                context.extend(u + [self.sep])
            context.pop()
            context = context[-self.args['max_len']:]
            response = response[:self.args['res_max_len']]
            ids = [self.cls] + context + [self.sep]
            rids = [self.cls] + response + [self.sep]
            return ids, rids
        else:
            data = item['data'][:-1]
            candidates = [item['data'][-1]] + item['neg_sentences']
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            context = []
            for u in items:
                context.extend(u + [self.sep])
            context.pop() 
            c = context[-self.args['max_len']:]
            c = [self.cls] + c + [self.sep]
            items = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']

            assert len(items) == 10
            rids, label = [], []
            for idx, utter in enumerate(items):
                if idx == 0:
                    label.append(1)
                else:
                    label.append(0)
                r = deepcopy(utter)
                r = r[:self.args['res_max_len']]
                ids_ = [self.cls] + c + [self.sep]
                rids_ = [self.cls] + r + [self.sep]
                rids.append(rids_)
            return c, rids, label

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [], []
            for a, b in batch:
                if a and b:
                    ids.append(torch.LongTensor(a))
                    rids.append(torch.LongTensor(b))
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
            }
        else:
            assert len(batch) == 1
            ids, rids, label = batch[0]
            ids = torch.LongTensor(ids)
            rids = pad_sequence([torch.LongTensor(i) for i in rids], batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, rids, rids_mask = to_cuda(ids, rids, rids_mask)
            label = label.cuda()
            return {
                'ids': ids,
                'rids': rids,
                'rids_mask': rids_mask,
                'label': label,
            }
