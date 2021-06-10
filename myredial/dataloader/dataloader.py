from header import *

def read_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            # context = ' [SEP] '.join(item['q']).strip()
            context = item['q']
            response = item['r'].strip()
            # NOTE: the candidates may not be 10 (less than 10)
            candidates = [i.strip() for i in item['nr'] if i.strip()]
            dataset.append((context, response, candidates))
            responses.extend([response] + candidates)
        responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} unique utterances in {path}')
    return dataset, responses


def read_text_data_fp(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            dataset.append((label, utterances))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data_hier(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = utterances[:-1], utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_response_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')[-1]
            if lang == 'zh':
                utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset


def read_context_data(path, lang='zh'):
    # also build the map from the context to response
    with open(path) as f:
        ctx, res = [], []
        for line in f.readlines():
            items = line.strip().split('\t')
            utterance = items[1:]
            label = items[0]
            if label == '0':
                continue
            if lang == 'zh':
                utterance = [''.join(u.split()) for u in utterance]
            context, response = utterance[:-1], utterance[-1]
            context = ' [SEP] '.join(context)
            ctx.append(context)
            res.append(response)
    print(f'[!] load {len(ctx)} context from {path}')
    return ctx, res


# ========== DUAL BERT HIERARCHICAL Dataset ========== #
class BERTDualHierarchicalDataset(Dataset):

    '''SET THE MAX LEN OF EACH UTTERANCE AS 64. The utterances that longer than 64 will be cut'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=64, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len

        # set hyperparameter in dataloader
        self.max_len = 64
        self.inner_bsz = 64

        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hier.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_hier(path, lang=lang)
        self.data = []
        if mode in ['train', 'train-post', 'train-dual-post']:
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
    
    def __init__(self, path, mode='train', lang='zh', max_len=300, model='bert-base-chinese', head=5, res_max_len=128):
        self.mode, self.max_len, self.res_max_len = mode, max_len, res_max_len
        self.head = head
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_one2many_{head}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        candidates = torch.load(f'{os.path.split(path)[0]}/candidates.pt')
        self.data = []
        if mode == 'train':
            data = read_text_data_one2many(path, lang=lang)
            responses = [r for c, r in data]
            for (context, response), cands in tqdm(list(zip(data, candidates))):
                cands = cands[:self.head-1]
                if len(cands) < self.head - 1:
                    cands.extend(random.sample(responses, self.head-1-len(cands)))
                item = self.vocab.batch_encode_plus([context, response] + cands)
                ids, rids = item['input_ids'][0], item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit(i, mode='res') for i in rids]
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
                
    def _length_limit(self, ids, mode='ctx'):
        if mode == 'ctx':
            if len(ids) > self.max_len:
                ids = [ids[0]] + ids[-(self.max_len-1):]
        else:
            if len(ids) > self.res_max_len:
                ids = ids[:self.res_max_len]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            # random sample head samples
            ids = torch.LongTensor(bundle['ids'])
            rids = [c for c in bundle['rids']]
            # rids = [rids[0]] + random.sample(rids[1:], self.head-1)
            rids = [torch.LongTensor(i) for i in rids]
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
class BERTDualInferenceContextResponseDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='inference', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference_ctx_res.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        context, response = read_context_data(path, lang=lang)
        self.data = []
        counter = 0
        for ctx, res in tqdm(list(zip(context, response))):
            item = self.vocab.encode(ctx)
            cids = self._length_limit_ctx(item)
            item = self.vocab.encode(res)
            rids = self._length_limit_res(item)
            self.data.append({
                'cid': cids, 'rid': rids, 'order': counter, 'cid_text': ctx, 'rid_text': res
            })
            counter += 1
                
    def _length_limit_ctx(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        if len(ids) > self.max_len:
            ids = ids[:self.max_len:]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        cid = torch.LongTensor(bundle['cid'])
        rid = torch.LongTensor(bundle['rid'])
        cid_text, rid_text = bundle['cid_text'], bundle['rid_text']
        order = bundle['order']
        return cid, rid, order, cid_text, rid_text

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
        cid = [i[0] for i in batch]
        rid = [i[1] for i in batch]
        order = [i[2] for i in batch]
        cid_text = [i[3] for i in batch]
        rid_text = [i[4] for i in batch]
        cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        cid_mask = self.generate_mask(cid)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            cid, rid, cid_mask, rid_mask = cid.cuda(), rid.cuda(), cid_mask.cuda(), rid_mask.cuda()
        return cid, cid_mask, rid, rid_mask, cid_text, rid_text, order


# ========== DUAL BERT INFERENCE CONTEXT Dataset ========== #
class BERTDualInferenceContextDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='inference', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference_ctx.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        context, response = read_context_data(path, lang=lang)
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
    
    def __init__(self, path, lang='zh', mode='inference', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_response_data(path, lang=lang)
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
class BERTDualWithNegDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='train', res_max_len=128, max_len=512, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.res_max_len = res_max_len 
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_writer.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data, responses = read_json_data(path, lang=lang)
        self.data = []
        if mode == 'train':
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                item = self.vocab.batch_encode_plus([context, response] + candidates)
                ids, rids = item['input_ids'][0], item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit_res(i) for i in rids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            for context, response, candidates in tqdm(data):
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                item = self.vocab.batch_encode_plus([context, response] + candidates)
                ids = item['input_ids'][0]
                rids = item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [1] + [0] * 10,
                    'ids': ids,
                    'rids': rids,
                })    
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.res_max_len:
            ids = ids[:self.res_max_len-1] + [self.sep]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
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
            rids = []
            for i in batch:
                rids.extend(i[1])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda()
            return ids, rids, ids_mask, rids_mask
        else:
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return ids, rids, rids_mask, label


# ========== BERT DUAL Dataset ========== #
class BERTDualFPDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='train', res_max_len=128, max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.res_max_len = res_max_len 
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        # NOTE:
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_fp.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        if mode == 'train':
            data = read_text_data_fp(path, lang=lang)
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                for i in range(1, len(utterances)):
                    context = ' [SEP] '.join(utterances[:i])
                    response = utterances[i]
                    item = self.vocab.batch_encode_plus([context, response])
                    ids, rids = item['input_ids'][0], item['input_ids'][1]
                    ids, rids = self._length_limit(ids), self._length_limit_res(rids)
                    self.data.append({
                        'ids': ids,
                        'rids': rids,
                    })
        else:
            data = read_text_data(path, lang=lang)
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus([item[1], item[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                ids, rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                })    
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.res_max_len:
            ids = ids[:self.res_max_len-1] + [self.sep]
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


# ========== BERT DUAL Dataset ========== #
class BERTDualDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='train', res_max_len=128, max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.res_max_len = res_max_len 
        # ipdb.set_trace()
        # self.vocab = PJBertTokenizer.from_pretrained('/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_base/data/vocab.txt')
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        # NOTE: for loading bert dual checkpoint
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if 'lccc' in path:
            data = read_text_data_fast(path)
            print(f'[!] fast dataloader activate ...')
        else:
            data = read_text_data(path, lang=lang)
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                ids, rids = self._length_limit(ids), self._length_limit_res(rids)
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
                ids, rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                })    
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.res_max_len:
            ids = ids[:self.res_max_len-1] + [self.sep]
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
    
    def __init__(self, path, mode='train', max_len=300, lang='zh', model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path, lang=lang)
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

class BERTWithNegDataset(Dataset):

    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
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


# ========== SABERT FT Dataset ========== #
class SABERTWithNegDataset(Dataset):

    '''segment embedding, token embedding, position embedding (default), mask embedding'''

    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_sa_neg.pt'
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
                    ids, tids, sids = self.annotate(utterances)
                    self.data.append({
                        'label': 1 if idx == 0 else 0, 
                        'ids': ids, 
                        'tids': tids, 
                        'sids': sids
                    })
        else:
            for context, response, candidates in tqdm(data):
                # we only need 10 candidates, pos:neg = 1:9
                # compatible with the douban, ecommerce, ubuntu-v1 corpus
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                ids, tids, sids = [],[], []
                for neg in [response] + candidates:
                    utterances = context + [neg]
                    ids_, tids_, sids_ = self.annotate(utterances)
                    ids.append(ids_)
                    tids.append(tids_)
                    sids.append(sids_)
                self.data.append({
                    'label': [1] + [0] * 9, 
                    'ids': ids, 
                    'tids': tids, 
                    'sids': sids,
                })

    def annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, sids, tcache, scache, l = ['[CLS]'], [0], [0], 0, 0, len(tokens)
        for idx, tok in enumerate(tokens):
            if idx < l - 1:
                ids.extend(tok)
                ids.append('[SEP]')
                tids.extend([tcache] * (len(tok) + 1))
                sids.extend([scache] * (len(tok) + 1))
                scache = 0 if scache == 1 else 1
                tcache = 0
            else:
                tcache = 1
                ids.extend(tok)
                tids.extend([tcache] * len(tok))
                sids.extend([scache] * len(tok))
        ids.append('[SEP]')
        tids.append(tcache)
        sids.append(scache)
        ids = self.vocab.encode(ids, add_special_tokens=False)
        ids, tids, sids = self._length_limit(ids), self._length_limit(tids), self._length_limit(sids)
        assert len(ids) == len(ids) and len(ids) == len(tids)
        return ids, tids, sids

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
            sids = torch.LongTensor(bundle['sids'])
            label = bundle['label']
            return ids, tids, sids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            sids = [torch.LongTensor(i) for i in bundle['sids']]
            return ids, tids, sids, bundle['label']

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
            ids, tids, sids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
        else:
            ids, tids, sids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                sids.extend(b[2])
                label.extend(b[3])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, sids, mask, label = ids.cuda(), tids.cuda(), sids.cuda(), mask.cuda(), label.cuda()
        return ids, tids, sids, mask, label


# ========== SABERT FT Dataset ========== #
class SABERTFTDataset(Dataset):

    '''segment embedding, token embedding, position embedding (default), mask embedding'''

    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_sa_ft.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_fp(path, lang=lang)
        self.data = []
        if mode == 'train':
            for label, utterances in tqdm(data):
                ids, tids, sids = self.annotate(utterances)
                self.data.append({'label': label, 'ids': ids, 'tids': tids, 'sids': sids})
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids_, tids_, sids_ = [], [], []
                for j in batch:
                    ids, tids, sids = self.annotate(j[1])
                    ids_.append(ids)
                    tids_.append(tids)
                    sids_.append(sids)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids_,
                    'tids': tids_,
                    'sids': sids_,
                })

    def annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, sids, tcache, scache, l = ['[CLS]'], [0], [0], 0, 0, len(tokens)
        for idx, tok in enumerate(tokens):
            if idx < l - 1:
                ids.extend(tok)
                ids.append('[SEP]')
                tids.extend([tcache] * (len(tok) + 1))
                sids.extend([scache] * (len(tok) + 1))
                scache = 0 if scache == 1 else 1
                tcache = 0
            else:
                tcache = 1
                ids.extend(tok)
                tids.extend([tcache] * len(tok))
                sids.extend([scache] * len(tok))
        ids.append('[SEP]')
        tids.append(tcache)
        sids.append(scache)
        ids = self.vocab.encode(ids, add_special_tokens=False)
        ids, tids, sids = self._length_limit(ids), self._length_limit(tids), self._length_limit(sids)
        assert len(ids) == len(ids) and len(ids) == len(tids)
        return ids, tids, sids

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
            sids = torch.LongTensor(bundle['sids'])
            label = bundle['label']
            return ids, tids, sids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            sids = [torch.LongTensor(i) for i in bundle['sids']]
            return ids, tids, sids, bundle['label']

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
            ids, tids, sids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
        else:
            # batch size is batch_size * 10
            ids, tids, sids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                sids.extend(b[2])
                label.extend(b[3])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, sids, mask, label = ids.cuda(), tids.cuda(), sids.cuda(), mask.cuda(), label.cuda()
        return ids, tids, sids, mask, label


# =========== load dataset ========== #
def load_dataset(args):
    DATASET_MAP = {
        'sa-bert-neg': SABERTWithNegDataset,
        'sa-bert': SABERTFTDataset,
        'bert-ft': BERTFTDataset,
        'dual-bert': BERTDualDataset,
        'dual-bert-gray': BERTDualWithNegDataset,
        'dual-bert-hierarchical-trs': BERTDualHierarchicalDataset,
    }

    INFERENCE_DATASET_MAP = {
        'dual-bert': (
            BERTDualInferenceDataset, 
            BERTDualInferenceContextDataset,
        )
    }

    # compatible for train, train-post, train-dual-post
    if args['mode'] in ['train', 'train-post', 'train-dual-post']:
        mode = 'train'
    else:
        mode = args['mode']

    path = f'{args["root_dir"]}/data/{args["dataset"]}/{mode}.txt'

    if mode == 'train':
        if args['model'] in ['bert-ft', 'sa-bert', 'sa-bert-neg', 'bert-ft-multi']:
            # Cross-encoder models doesn't need res_max_len parameters
            data = DATASET_MAP[args['model']](path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        else:
            data = DATASET_MAP[args['model']](path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'], res_max_len=args['res_max_len'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        sampler = train_sampler
    elif mode == 'inference_qa':
        path = f'data/{args["dataset"]}/train.txt'
        data = BERTDualInferenceContextResponseDataset(path, lang=args['lang'], mode=mode, max_len=args['max_len'], model=args['pretrained_model'])
        sampler = torch.utils.data.distributed.DistributedSampler(
            data,
           num_replicas=dist.get_world_size(),
            rank=args['local_rank'],
        )
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
        sampler = None
    elif mode == 'inference':
        # only inference train dataset
        path = f'data/{args["dataset"]}/train.txt'
        data_res = INFERENCE_DATASET_MAP[args['model']][0](path, lang=args['lang'], mode=mode, max_len=args['max_len'], model=args['pretrained_model'])
        data_ctx = INFERENCE_DATASET_MAP[args['model']][1](path, lang=args['lang'], mode=mode, max_len=args['max_len'], model=args['pretrained_model'])

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
        sampler = None
    else:
        data = DATASET_MAP[args['model']](path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
        iter_ = DataLoader(data, batch_size=args['test_batch_size'], collate_fn=data.collate)
        sampler = None

    if mode == 'inference':
        if not os.path.exists(data_ctx.pp_path):
            data_ctx.save()
        if not os.path.exists(data_res.pp_path):
            data_res.save()
    elif mode == 'inference_qa':
        if not os.path.exists(data.pp_path):
            data.save()
    else:
        if not os.path.exists(data.pp_path):
            data.save()
    return data, iter_, sampler
