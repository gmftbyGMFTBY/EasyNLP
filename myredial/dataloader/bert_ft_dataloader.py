from header import *
from .utils import *


class BERTFTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                context = ' [SEP] '.join(utterances[:-1])
                response = utterances[-1]
                item = self.vocab.batch_encode_plus([[context, response]])
                ids = item['input_ids'][0]
                tids = item['token_type_ids'][0]
                ids, tids = self._length_limit(ids), self._length_limit(tids)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                d_ = []
                for b in batch:
                    context = ' [SEP] '.join(b[1][:-1])
                    response = b[1][-1]
                    d_.append([context, response])
                item = self.vocab.batch_encode_plus(d_)
                ids = item['input_ids']
                tids = item['token_type_ids']
                ids, tids = [self._length_limit(ids_) for ids_ in ids], [self._length_limit(tids_) for tids_ in tids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                })    
                
    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            label = bundle['label']
            return ids, tids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, tids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
        else:
            # batch size is batch_size * 10
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, mask, label = ids.cuda(), tids.cuda(), mask.cuda(), label.cuda()
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask, 
            'label': label
        }

class BERTFTWithNegDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_neg_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            if self.args['mode'] == 'train':
                self.extract_by_gray_num()
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # data, responses = read_json_data(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            data, responses = read_text_data_with_neg_q_r_neg(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context)
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                item = self.vocab.batch_encode_plus([
                    [context, res] for res in [response] + candidates
                ])
                ids = item['input_ids']
                tids = item['token_type_ids']
                ids = self._length_limit(ids)
                tids = self._length_limit(tids)
                self.data.append({
                    'label': [1] + [0] * 10, 
                    'ids': ids, 
                    'tids': tids, 
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context)
                # we only need 10 candidates, pos:neg = 1:9
                # compatible with the douban, ecommerce, ubuntu-v1 corpus
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                item = self.vocab.batch_encode_plus([
                    [context, res] for res in [response] + candidates
                ])
                ids = item['input_ids']
                tids = item['token_type_ids']
                ids = self._length_limit(ids)
                tids = self._length_limit(tids)
                self.data.append({
                    'label': [1] + [0] * 9, 
                    'ids': ids, 
                    'tids': tids, 
                })

    def extract_by_gray_num(self):
        # process self.data (after loaded)
        num = self.args['gray_cand_num']
        dataset = []
        for sample in tqdm(self.data):
            dataset.append({
                'label': 1,
                'ids': sample['ids'][0],
                'tids': sample['tids'][0]
            })
            # neg
            neg_idx = random.sample(range(1, 11), num)
            neg_ids = [sample['ids'][i] for i in neg_idx]
            neg_tids = [sample['tids'][i] for i in neg_idx]
            for i, j in zip(neg_ids, neg_tids):
                dataset.append({
                    'label': 0,
                    'ids': i,
                    'tids': j,
                })
        self.data = dataset

    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            try:
                ids = torch.LongTensor(bundle['ids'])
            except:
                ipdb.set_trace()
            tids = torch.LongTensor(bundle['tids'])
            return ids, tids, bundle['label']
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, tids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')

    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask

    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            tids = [i[1] for i in batch]
            label = [i[2] for i in batch]
        else:
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, mask, label = ids.cuda(), tids.cuda(), mask.cuda(), label.cuda()
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask, 
            'label': label
        }
