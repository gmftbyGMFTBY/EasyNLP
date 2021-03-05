from header import *
from model import MemoryBank


def read_text_data(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            utterances = [''.join(u.split()) for u in utterances]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data_fast(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data_one2many(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if label == 0:
                continue
            utterances = [''.join(u.split()) for u in utterances]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data_hier(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            utterances = [''.join(u.split()) for u in utterances]
            context, response = utterances[:-1], utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_response_data(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')[-1]
            utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset


def read_context_data(path):
    # also build the map from the context to response
    with open(path) as f:
        ctx, res = [], []
        for line in f.readlines():
            items = line.strip().split('\t')
            utterance = items[1:]
            label = items[0]
            if label == '0':
                continue
            utterance = [''.join(u.split()) for u in utterance]
            context, response = utterance[:-1], utterance[-1]
            context = ' [SEP] '.join(context)
            ctx.append(context)
            res.append(response)
    print(f'[!] load {len(ctx)} responses from {path}')
    return ctx, res


# ========== DUAL BERT HIERARCHICAL Dataset ========== #
class BERTDualHierarchicalDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.inner_bsz = 32 
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hier.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_hier(path)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(context + [response])
                cids, rids = item['input_ids'][:-1], item['input_ids'][-1]
                cids, rids = [self._length_limit(ids) for ids in cids], self._length_limit(rids)
                self.data.append({
                    'cids': cids,
                    'rids': rids,
                    'cids_turn_length': len(cids)
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus(item[1] + [item[2]])
                    cids = item['input_ids'][:-1]
                    rids.append(item['input_ids'][-1])
                cids, rids = [self._length_limit(ids) for ids in cids], [self._length_limit(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'cids': cids,
                    'rids': rids,
                    'cids_turn_length': len(cids)
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
            cids = [torch.LongTensor(i) for i in bundle['cids']]
            rids = torch.LongTensor(bundle['rids'])
            cids_turn_length = bundle['cids_turn_length']
            return cids, rids, cids_turn_length
        else:
            cids = [torch.LongTensor(i) for i in bundle['cids']]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            cids_turn_length = bundle['cids_turn_length']
            return cids, rids, cids_turn_length, bundle['label']

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
        '''for training procedure, bigger batch size lead to longer processing time, because of the more useless padding tokens; we use the inner bsz to resort method to reduce the number of the padding tokens'''
        if self.mode == 'train':
            rids, cids_turn_length = [i[1] for i in batch], [i[2] for i in batch]
            cids = []
            for i in batch:
                cids.extend(i[0])
            # count the length
            lengths = [len(i) for i in cids]
            lengths_order = np.argsort(lengths)
            cids = [cids[i] for i in lengths_order]
            recover_mapping = {i:idx for idx, i in enumerate(lengths_order)}

            chunks = [cids[i:i+self.inner_bsz] for i in range(0, len(lengths), self.inner_bsz)]
            cids = [pad_sequence(item, batch_first=True, padding_value=self.pad).cuda() for item in chunks]
            cids_mask = [self.generate_mask(item).cuda() for item in cids]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                rids, rids_mask = rids.cuda(), rids_mask.cuda()
            return cids, rids, cids_turn_length, cids_mask, rids_mask, recover_mapping
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            cids, rids, cids_turn_length, label = batch[0], batch[1], batch[2], batch[3]
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            cids_mask = self.generate_mask(cids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                cids, rids, cids_mask, rids_mask, label = cids.cuda(), rids.cuda(), cids_mask.cuda(), rids_mask.cuda(), label.cuda()
            return cids, rids, cids_turn_length, cids_mask, rids_mask, label


# ========== DUAL BERT ONE2MANY Dataset ========== #
class BERTDualOne2ManyDataset(Dataset):
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese', head=5):
        self.mode, self.max_len = mode, max_len
        self.head = head
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_one2many_{head}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        candidates = torch.load(f'{os.path.split(path)[0]}/candidates.pt')
        self.data = []
        if mode == 'train':
            data = read_text_data_one2many(path)
            for (context, response), cands in tqdm(list(zip(data, candidates))):
                # cands = cands[:self.head-1]
                cands = random.sample(cands, self.head-1)
                item = self.vocab.batch_encode_plus([context, response] + cands)
                ids, rids = item['input_ids'][0], item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit(i) for i in rids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            data = read_text_data(path)
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus([item[1], item[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                ids, rids = self._length_limit(ids), [self._length_limit(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
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
            rids = [torch.LongTensor(c) for c in bundle['rids']]
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

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
            ids = [i[0] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            
            rids, rids_mask = [], []
            num_cand = len(batch[0][1])
            for i in range(num_cand):
                cands = [item[1][i] for item in batch]
                cids_ = pad_sequence(cands, batch_first=True, padding_value=self.pad)
                cids_mask_ = self.generate_mask(cids_)
                if torch.cuda.is_available():
                    cids_, cids_mask_ = cids_.cuda(), cids_mask_.cuda()
                rids.append(cids_)
                rids_mask.append(cids_mask_)
            if torch.cuda.is_available():
                ids, ids_mask = ids.cuda(), ids_mask.cuda()
            return ids, rids, ids_mask, rids_mask
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return ids, rids, rids_mask, label


# ========== DUAL BERT INFERENCE CONTEXT Dataset ========== #
class BERTDualInferenceContextDataset(Dataset):
    
    def __init__(self, path, mode='inference', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference_ctx.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        context, response = read_context_data(path)
        self.data = []
        counter = 0
        for ctx, res in tqdm(list(zip(context, response))):
            item = self.vocab.encode(ctx)
            ids = self._length_limit(item)
            self.data.append({'ids': ids, 'ctx_text': ctx, 'res_text': res, 'order': counter})
            counter += 1
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        ctx_text = bundle['ctx_text']
        res_text = bundle['res_text']
        order = bundle['order']
        return ids, ctx_text, res_text, order

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
        ids = [i[0] for i in batch]
        ctx_text = [i[1] for i in batch]
        res_text = [i[2] for i in batch]
        order = [i[3] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, ids_mask = ids.cuda(), ids_mask.cuda()
        return ids, ids_mask, ctx_text, res_text, order


# ========== DUAL BERT INFERENCE Dataset ============ #
class BERTDualInferenceDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, mode='inference', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_response_data(path)
        self.data = []
        for response in tqdm(data):
            item = self.vocab.encode(response)
            ids = self._length_limit(item)
            self.data.append({'ids': ids, 'text': response})
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        text = bundle['text']
        return ids, text

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
        ids = [i[0] for i in batch]
        text = [i[1] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, ids_mask = ids.cuda(), ids_mask.cuda()
        return ids, ids_mask, text

# ========== BERT DUAL Dataset ========== #
class BERTDualDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if 'lccc' in path:
            data = read_text_data_fast(path)
            print(f'[!] fast dataloader activate ...')
        else:
            data = read_text_data(path)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                ids, rids = self._length_limit(ids), self._length_limit(rids)
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus([item[1], item[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                ids, rids = self._length_limit(ids), [self._length_limit(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
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
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

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
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda()
            return ids, rids, ids_mask, rids_mask
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return ids, rids, rids_mask, label


# ========== BERT FT Dataset ========== # 
class BERTFTDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
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

# ========== BERT Generation ========== #
class BERTGenDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_gen.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([[context, response]])
                ids = item['input_ids'][0]
                tids = item['token_type_ids'][0]
                ids, tids = self._length_limit(ids), self._length_limit(tids)
                label = torch.where(tids == 1, ids, torch.LongTensor([0] * len(ids)))[1:]
                self.data.append({ 
                    'label': label,
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                item = data[i]
                item = self.vocab.batch_encode_plus([item[1], item[2]])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                tids = item['token_type_ids'][0]
                ids, tids = self._length_limit(ids), self._length_limit(tids)
                rids = self._length_limit(rids)
                self.data.append({
                    'ids': ids,
                    'tids': tids,
                    'rids': rids
                })    
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return torch.LongTensor(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            return bundle['ids'], bundle['tids'], bundle['label']
        else:
            return bundle['ids'], bundle['tids'], bundle['rids']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        '''similar to UniLM
        https://github.com/huggingface/transformers/issues/9366
        context bi-direction self-attention and response single direction attention
        
        :ids [B, S]: is the token type ids: context / response / padding
        for example,
        [
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        => 3D attention mask
        '''
        length = ids.shape[1]
        attention_mask_3d = []
        # attend to encoder and cut the padding
        for ids_ in ids:
            mask = torch.full((length, length), 0)
            mask_cond = torch.arange(mask.size(-1))
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
            # context length and response length
            nonzero_index = ids_.nonzero().squeeze()
            x, y = nonzero_index[0].item(), nonzero_index[-1].item() + 1
            mask[:, :x] = 1
            mask[y:, :] = 0
            attention_mask_3d.append(mask)
        attention_mask_3d = torch.stack(attention_mask_3d)
        return attention_mask_3d
        
    def collate(self, batch):
        if self.mode == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            label = pad_sequence(label, batch_first=True, padding_value=self.pad)
            mask = self.generate_mask(tids)
        else:
            # batch size is 1
            batch = batch[0]
            ids, tids, label = [batch[0]], [batch[1]], [batch[2]]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            label = pad_sequence(label, batch_first=True, padding_value=self.pad)
            length = ids.shape[1]
            mask = torch.full((length, length), 1).view(1, length, length).long()
        if torch.cuda.is_available():
            ids, tids, mask, label = ids.cuda(), tids.cuda(), mask.cuda(), label.cuda()
        # ids: [B, S], tids: [B, S], mask: [B, S, S], label: [B, S]
        return ids, tids, mask, label

# ========== BERT Generation and Classification ========== #
class BERTGenFTDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding
    
    Only employ classification inference during test mode'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_gen_ft.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                item = self.vocab.batch_encode_plus([[context, response]])
                ids = item['input_ids'][0]
                tids = item['token_type_ids'][0]
                ids, tids = self._length_limit(ids), self._length_limit(tids)
                if label == 1:
                    lm_label = torch.where(tids == 1, ids, torch.LongTensor([0] * len(ids)))[1:]
                else:
                    lm_label = torch.LongTensor([0] * len(ids))[1:]
                self.data.append({ 
                    'lm_label': lm_label,
                    'cls_label': label,
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
                    'cls_label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                })  
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return torch.LongTensor(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            return bundle['ids'], bundle['tids'], bundle['cls_label'], bundle['lm_label']
        else:
            return bundle['ids'], bundle['tids'], bundle['cls_label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        '''similar to UniLM
        https://github.com/huggingface/transformers/issues/9366
        context bi-direction self-attention and response single direction attention
        
        :ids [B, S]: is the token type ids: context / response / padding
        return the 3D attention mask'''
        length = ids.shape[1]
        attention_mask_3d = []
        # attend to encoder and cut the padding
        for ids_ in ids:
            mask = torch.full((length, length), 0)
            mask_cond = torch.arange(mask.size(-1))
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
            # context length and response length
            nonzero_index = ids_.nonzero().squeeze(-1)
            x, y = nonzero_index[0].item(), nonzero_index[-1].item() + 1
            mask[:, :x] = 1
            mask[y:, :] = 0
            attention_mask_3d.append(mask)
        attention_mask_3d = torch.stack(attention_mask_3d)
        return attention_mask_3d
        
    def collate(self, batch):
        if self.mode == 'train':
            ids, tids, cls_label, lm_label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            lm_label = pad_sequence(lm_label, batch_first=True, padding_value=self.pad)
            cls_label = torch.LongTensor(cls_label)
            mask = self.generate_mask(tids)
            if torch.cuda.is_available():
                ids, tids, mask, lm_label, cls_label = ids.cuda(), tids.cuda(), mask.cuda(), lm_label.cuda(), cls_label.cuda()
            # ids: [B, S], tids: [B, S], mask: [B, S, S], label: [B, S]
            return ids, tids, mask, cls_label, lm_label
        else:
            # batch size is batch_size *10
            ids, tids, cls_label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                cls_label.extend(b[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            cls_label = torch.LongTensor(cls_label)
            mask = self.generate_mask(tids)
            if torch.cuda.is_available():
                ids, tids, mask, cls_label = ids.cuda(), tids.cuda(), mask.cuda(), cls_label.cuda()
            # ids: [B, S], tids: [B, S], mask: [B, S, S], label: [B, S]
            return ids, tids, mask, cls_label


# ========== BERT DUAL with Memory Bank Dataset ========== #
class BERTDualMBDataset(Dataset):

    '''init and save the memory bank'''
    
    def __init__(self, path, mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_mb.pt'
        if os.path.exists(self.pp_path):
            _, self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if 'lccc' in path:
            data = read_text_data_fast(path)
            print(f'[!] fast dataloader activate ...')
        else:
            data = read_text_data(path)
        self.data = []
        if mode == 'train':
            # NOTE: MEMORY BANK
            counter, corpus = 0, {}
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                ids, rids = self._length_limit(ids), self._length_limit(rids)
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'id': counter
                })
                corpus[counter] = response
                counter += 1
            self.mb = MemoryBank(corpus)
        else:
            self.mb = None
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus([item[1], item[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                ids, rids = self._length_limit(ids), [self._length_limit(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
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
            rids = torch.LongTensor(bundle['rids'])
            idx = bundle['id']
            return idx, ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        torch.save((self.mb, self.data), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.mode == 'train':
            idx, ids, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda()
            return idx, ids, rids, ids_mask, rids_mask
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return ids, rids, rids_mask, label
    

# =========== load dataset ========== #
def load_dataset(args):
    DATASET_MAP = {
        'bert-gen': BERTGenDataset,
        'bert-ft': BERTFTDataset,
        'bert-gen-ft': BERTGenFTDataset,
        'dual-bert': BERTDualDataset,
        'dual-bert-mb': BERTDualMBDataset,
        'dual-bert-poly': BERTDualDataset,
        'dual-bert-cl': BERTDualDataset,
        'dual-bert-vae': BERTDualDataset,
        'dual-bert-vae2': BERTDualDataset,
        'dual-bert-one2many': BERTDualOne2ManyDataset,
        'dual-bert-hierarchical': BERTDualHierarchicalDataset,
    }

    INFERENCE_DATASET_MAP = {
        'dual-bert': (
            BERTDualInferenceDataset, 
            BERTDualInferenceContextDataset,
        )
    }
    
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] == 'train':
        if args['model'] == 'dual-bert-one2many':
            data = DATASET_MAP[args['model']](path, mode=args['mode'], max_len=args['max_len'], model=args['pretrained_model'], head=args['head_num'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        else:
            data = DATASET_MAP[args['model']](path, mode=args['mode'], max_len=args['max_len'], model=args['pretrained_model'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
    elif args['mode'] == 'inference':
        # only inference train dataset
        path = f'data/{args["dataset"]}/train.txt'
        data_res = INFERENCE_DATASET_MAP[args['model']][0](path, mode=args['mode'], max_len=args['max_len'], model=args['pretrained_model'])
        data_ctx = INFERENCE_DATASET_MAP[args['model']][1](path, mode=args['mode'], max_len=args['max_len'], model=args['pretrained_model'])

        res_sampler = torch.utils.data.distributed.DistributedSampler(
            data_res,
            num_replicas=dist.get_world_size(),
            rank=args['local_rank'],
        )
        iter_res = DataLoader(data_res, batch_size=args['batch_size'], collate_fn=data_res.collate, sampler=res_sampler)

        ctx_sampler = torch.utils.data.distributed.DistributedSampler(
            data_ctx,
            num_replicas=dist.get_world_size(),
            rank=args['local_rank'],
        )
        iter_ctx = DataLoader(data_ctx, batch_size=args['batch_size'], collate_fn=data_ctx.collate, sampler=ctx_sampler)

        iter_ = (iter_res, iter_ctx)
        data = (data_res, data_ctx)
    else:
        data = DATASET_MAP[args['model']](path, mode=args['mode'], max_len=args['max_len'], model=args['pretrained_model'])
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
    if args['mode'] == 'inference':
        if not os.path.exists(data_ctx.pp_path):
            data_ctx.save()
        if not os.path.exists(data_res.pp_path):
            data_res.save()
    else:
        if not os.path.exists(data.pp_path):
            data.save()
    return data, iter_


if __name__ == "__main__":
    # ========== BERTFTDataset ========== #
    train_data = BERTFTDataset('data/ubuntu/train.txt', mode='train')
    train_iter = DataLoader(train_data, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTFTDataset ========== #
    # ========== BERTGenDataset ========== #
    # train_data = BERTGenDataset('data/ecommerce/train.txt', mode='train')
    # train_iter = DataLoader(train_data, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTGenDataset ========== #
    # ========== BERTGenFTDataset ========== #
    # train_data = BERTGenFTDataset('data/ubuntu/train.txt', mode='train')
    # train_iter = DataLoader(train_data, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTGenFTDataset ========== #
    train_data.save()
    for batch in tqdm(train_iter):
        ipdb.set_trace()
