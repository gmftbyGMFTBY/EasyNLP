from header import *


class BERTFTDataset(Dataset):
    
    def __init__(self, vocab, path, mode='train', max_len=300, lang='zh'):
        self.mode, self.max_len = mode, max_len
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_utterances(path, lang=lang)
        self.data = []
        if mode == 'train':
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
                item = self.vocab.batch_encode_plus([[b[1], b[2]] for b in batch])
                ids = item['input_ids']
                tids = item['token_type_ids']
                ids, tids = [self._length_limit(ids_) for ids_ in ids], [self._length_limit(tids_) for tids_ in tids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                })    
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
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
        if self.mode == 'train':
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
        # ids: [B, S]
        # tids: [B, S]
        # mask: [B, S]
        return ids, tids, mask, label

class BERTWithNegDataset(Dataset):

    def __init__(self, vocab, path, lang='zh', mode='train', max_len=300):
        self.mode, self.max_len = mode, max_len
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_neg.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data, responses = read_json_data(path, lang=lang)
        self.data = []
        if mode == 'train':
            for context, response, candidates in tqdm(data):
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                for idx, neg in enumerate([response] + candidates):
                    utterances = context + [neg]
                    ids, tids = self.annotate(utterances)
                    self.data.append({
                        'label': 1 if idx == 0 else 0, 
                        'ids': ids, 
                        'tids': tids, 
                    })
        else:
            for context, response, candidates in tqdm(data):
                # we only need 10 candidates, pos:neg = 1:9
                # compatible with the douban, ecommerce, ubuntu-v1 corpus
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                ids, tids = [], []
                for neg in [response] + candidates:
                    utterances = context + [neg]
                    ids_, tids_ = self.annotate(utterances)
                    ids.append(ids_)
                    tids.append(tids_)
                self.data.append({
                    'label': [1] + [0] * 9, 
                    'ids': ids, 
                    'tids': tids, 
                })

    def annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, tcache, l = ['[CLS]'], [0], 0, len(tokens)
        for idx, tok in enumerate(tokens):
            if idx < l - 1:
                ids.extend(tok)
                ids.append('[SEP]')
                tids.extend([tcache] * (len(tok) + 1))
                tcache = 0
            else:
                tcache = 1
                ids.extend(tok)
                tids.extend([tcache] * len(tok))
        ids.append('[SEP]')
        tids.append(tcache)
        ids = self.vocab.encode(ids, add_special_tokens=False)
        assert len(ids) == len(tids)
        ids, tids = self._length_limit(ids), self._length_limit(tids)
        return ids, tids

    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
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
        if self.mode == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
        else:
            ids, tids, sids, label = [], [], [], []
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
        return ids, tids, mask, label
