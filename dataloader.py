from header import *


class InputExamples:
    
    def __init__(self, utterances, response, label, seq_lengths):
        self.utterances = utterances
        self.response = response
        self.label = label
        self.dialog_len, self.response_len = seq_lengths


def read_ums_data(path, vocab):
    with open(path) as f:
        data = [line.strip() for line in f if len(line.strip()) > 0]
    dataset = []
    for dialog in tqdm(dta):
        dialog_data = dialog.split('\t')
        label = dialog_data[0]
        utterances, dialog_len = [], []
        for utt in dialog_data[1:-1]:
            utt_tok = vocab.tokenize(utt)
            utterances.append(utt_tok)
            dialog_len.append(len(utt_tok))
        response = vocab.tokenize(dialog_data[-1])
        dataset.append(
            InputExamples(
                utterances=utterances,
                response=response,
                label=int(label),
                seq_lengths=(dialog_len, len(response))
            )
        )
    print(f'[!] collect {len(dataset)} samples')
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


def read_text_data_sa(path, lang='zh'):
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


def read_text_data_one2many(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if label == 0:
                continue
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((context, response))
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


def read_text_data_hier_gru(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
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


# ========== DUAL GRU HIERARCHICAL Dataset ========== #
class GRUDualHierarchicalDataset(Dataset):

    def __init__(self, path, vocab_path='', lang='zh', mode='train', max_len=64, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len

        # set hyperparameter in dataloader
        self.max_len = 64
        self.inner_bsz = 64

        # load vocab from the pretrained language model
        vocab, _ = torch.load(vocab_path)
        self.vocab = {word:idx for idx, word in enumerate(vocab)}
        self.pad = self.vocab['[PAD]']
        self.unk = self.vocab['[UNK]']
        self.sos = self.vocab['[SOS]']
        self.eos = self.vocab['[EOS]']

        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gru_hier.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_hier_gru(path, lang=lang)
        self.data = []
        if mode in ['train']:
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                # text to ids
                cids = [self.encode(u) for u in context]
                rids = self.encode(response)
                cids, rids = [self._length_limit(ids) for ids in cids], self._length_limit(rids)
                self.data.append({
                    'cids': cids,
                    'rids': rids,
                    'cids_turn_length': len(cids),
                    'cids_length': [len(i) for i in cids],
                    'rids_length': len(rids)
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    # text to ids
                    cids = [self.encode(u) for u in item[1]]
                    rids.append(self.encode(item[2]))
                cids, rids = [self._length_limit(ids) for ids in cids], [self._length_limit(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'cids': cids,
                    'rids': rids,
                    'cids_turn_length': len(cids),
                    'cids_length': [len(i) for i in cids],
                    'rids_length': [len(i) for i in rids]
                })    
    
    def encode(self, utterance):
        ids = [self.sos] + [self.vocab[word] if word in self.vocab else self.vocab['[UNK]'] for word in utterance.split()] + [self.eos]
        return ids
                
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
            cids_length = bundle['cids_length']
            rids_length = bundle['rids_length']
            return cids, rids, cids_turn_length, cids_length, rids_length
        else:
            cids = [torch.LongTensor(i) for i in bundle['cids']]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            cids_turn_length = bundle['cids_turn_length']
            cids_length = bundle['cids_length']
            rids_length = bundle['rids_length']
            return cids, rids, cids_turn_length, cids_length, rids_length, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.mode == 'train':
            rids, cids_turn_length = [i[1] for i in batch], [i[2] for i in batch]
            cids_length, cids = [], []
            rids_length = [i[4] for i in batch]
            for i in batch:
                cids.extend(i[0])
                cids_length.extend(i[3])
            # count the length
            lengths = [len(i) for i in cids]
            lengths_order = np.argsort(lengths)
            cids = [cids[i] for i in lengths_order]
            cids_length = [cids_length[i] for i in lengths_order]
            recover_mapping = {i:idx for idx, i in enumerate(lengths_order)}

            chunks = [cids[i:i+self.inner_bsz] for i in range(0, len(lengths), self.inner_bsz)]
            cids = [pad_sequence(item, batch_first=True, padding_value=self.pad).cuda() for item in chunks]
            cids_length = [cids_length[i:i+self.inner_bsz] for i in range(0, len(lengths), self.inner_bsz)]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_length = torch.LongTensor(rids_length)
            if torch.cuda.is_available():
                rids, rids_length = rids.cuda(), rids_length.cuda()
            return cids, rids, cids_turn_length, cids_length, rids_length, recover_mapping
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            cids, rids, cids_turn_length, cids_length, rids_length, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                cids, rids, label = cids.cuda(), rids.cuda(), label.cuda()
            return cids, rids, cids_turn_length, cids_length, rids_length, label


# ========== DUAL BERT HIERARCHICAL Dataset ========== #
class BERTDualHierarchicalKDDataset(Dataset):

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
            for (context, response), cands in tqdm(list(zip(data, candidates))):
                # cands = cands[:self.head-1]
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
            rids = [rids[0]] + random.sample(rids[1:], self.head-1)
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
            cids = self._length_limit_ctx(item)
            item = self.vocab.encode(res)
            rids = self._length_limit_res(item)
            self.data.append({'cid': cids, 'rid': rids, 'order': counter})
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
        order = bundle['order']
        return cid, rid, order

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
        cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        cid_mask = self.generate_mask(cid)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            cid, rid, cid_mask, rid_mask = cid.cuda(), rid.cuda(), cid_mask.cuda(), rid_mask.cuda()
        return cid, cid_mask, rid, rid_mask, order


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


class BERTDualGenDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, res_max_len=64, model='bert-base-chinese'):
        self.mode, self.max_len, self.res_max_len = mode, max_len, res_max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gen.pt'
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
                item = self.vocab.batch_encode_plus([[context, response], response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                tids = item['token_type_ids'][0]
                ids, rids = self._length_limit(ids), self._length_limit_res(rids)
                tids = self._length_limit(tids)
                label = torch.where(tids == 1, ids, torch.LongTensor([0] * len(ids)))[1:]
                self.data.append({
                    'ids': ids,
                    'tids': tids,
                    'rids': rids,
                    'label': label
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
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return torch.LongTensor(ids)
    
    def _length_limit_res(self, ids):
        if len(ids) > self.res_max_len:
            ids = ids[:self.res_max_len]
        return torch.LongTensor(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            return bundle['ids'], bundle['tids'], bundle['rids'], bundle['label']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return bundle['ids'], bundle['rids'], bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask

    def generate_mask_(self, ids):
        length = ids.shape[1]
        attention_mask_3d = []
        for ids_ in ids:
            mask = torch.full((length, length), 0)
            mask_cond = torch.arange(mask.size(-1))
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
            nonzero_index = ids_.nonzero().squeeze()
            x, y = nonzero_index[0].item(), nonzero_index[-1].item() + 1
            mask[:, :x] = 1
            mask[y:, :] = 0
            attention_mask_3d.append(mask)
        attention_mask_3d = torch.stack(attention_mask_3d)
        return attention_mask_3d
        
    def collate(self, batch):
        if self.mode == 'train':
            ids, rids = [i[0] for i in batch], [i[2] for i in batch]
            tids = [i[1] for i in batch]
            label = [i[3] for i in batch]
            label = pad_sequence(label, batch_first=True, padding_value=self.pad)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask_(tids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, tids, rids, ids_mask, rids_mask, label = ids.cuda(), tids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda(), label.cuda()
            return ids, tids, rids, ids_mask, rids_mask, label
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            ids = ids.unsqueeze(0)    # [1, S]
            tids = torch.zeros_like(ids)    # [1, S]
            ids_mask = torch.ones(1, ids.shape[-1], ids.shape[-1]).cuda()
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, ids_mask, tids, rids, rids_mask, label = ids.cuda(), ids_mask.cuda(), tids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return ids, ids_mask, tids, rids, rids_mask, label


# ========== BERT DUAL Dataset ========== #
class BERTDualCLDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.ctx = self.vocab.convert_tokens_to_ids('[CTX]')
        self.res = self.vocab.convert_tokens_to_ids('[RES]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_cl.pt'
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
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
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
class BERTDualMLMDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.mlm_prob = 0.15
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_mlm.pt'
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
                (ids, labels), (rids, rids_labels) = self.transform_mlm(ids), self.transform_mlm(rids)
                self.data.append({
                    'ids': ids,
                    'labels': labels,
                    'rids': rids,
                    'rids_labels': rids_labels,
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

    def transform_mlm(self, ids):
        '''after limitation, only for training'''
        # ignore the [SEP], [CLS] tokens
        valid_tokens = [idx for idx, i in enumerate(ids) if i not in [self.cls, self.sep]]
        cands = random.sample(valid_tokens, int(self.mlm_prob * len(valid_tokens)))
        tokens, labels = [], []
        for idx in range(len(ids)):
            if idx in cands:
                if random.random() < 0.8:
                    token = self.mask
                elif random.random() < 0.5:
                    token = ids[idx]
                else:
                    token = random.randint(0, len(self.vocab) - 1)
                tokens.append(token)
                labels.append(ids[idx])
            else:
                tokens.append(ids[idx])
                labels.append(self.pad)    # ignored in mlm loss
        return tokens, labels
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            ids = torch.LongTensor(bundle['ids'])
            ids_labels = torch.LongTensor(bundle['labels'])
            rids = torch.LongTensor(bundle['rids'])
            rids_labels = torch.LongTensor(bundle['rids_labels'])
            return ids, ids_labels, rids, rids_labels
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
            ids, ids_labels, rids, rids_labels = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_labels = pad_sequence(ids_labels, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_labels = pad_sequence(rids_labels, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask, ids_labels, rids_labels = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda(), ids_labels.cuda(), rids_labels.cuda()
            return ids, rids, ids_mask, rids_mask, ids_labels, rids_labels
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
class BERTDualCurriculumDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.res_max_len = max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_curriculum.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path, lang=lang)
        self.corr_matrix = torch.load(f'{os.path.split(path)[0]}/corr_matrix.pt')
        print(f'[!] load corr matrix over')
        self.data = []
        if mode == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                (ids, sids, speaker), rids = self._length_limit(ids), self._length_limit_res(rids)
                self.data.append({
                    'ids': ids,
                    's_ids': sids,
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
                (ids, sids, speaker), rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    's_ids': sids,
                    'rids': rids,
                })    
        # curriculum learning
        self.step_counter = 0
        self.total_step = None
        self.min_space = 1000
        self.bsz = None
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        speaker = 0
        sids = []
        for i in ids:
            sids.append(speaker)
            if i == self.sep:
                speaker = 1 if speaker == 0 else 0
        return ids, sids, speaker
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.res_max_len:
            ids = ids[:self.res_max_len-1] + [self.sep]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # also return the hard negative samples
        bundle = self.data[i]
        self.max_space = int(len(self.data) * self.step_counter / self.total_step)
        self.max_space = max(self.min_space, self.max_space)
        sim = torch.sort(self.corr_matrix[i])[1][-self.max_space:]
        sim = random.sample(sim.tolist(), self.bsz)
        if self.mode == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            s_ids = torch.LongTensor(bundle['s_ids'])
            hard_rids = [torch.LongTensor(self.data[i]['rids']) for i in sim]
            return ids, rids, s_ids, hard_rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            s_ids = torch.LongTensor(bundle['s_ids'])
            return ids, rids, s_ids, bundle['label']

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
            s_ids = [i[2] for i in batch]
            hard_rids = []
            for i in batch:
                hard_rids.extend(i[3])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            hard_rids = pad_sequence(hard_rids, batch_first=True, padding_value=self.pad)
            s_ids = pad_sequence(s_ids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            hard_rids_mask = self.generate_mask(hard_rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask, s_ids, hard_rids, hard_rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda(), s_ids.cuda(), hard_rids.cuda(), hard_rids_mask.cuda()
            self.step_counter += 1
            return ids, rids, ids_mask, rids_mask, s_ids, hard_rids, hard_rids_mask
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, s_ids, label = batch[0], batch[1], batch[2], batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, s_ids, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), s_ids.cuda(), label.cuda()
            return ids, rids, rids_mask, s_ids, label



# ========== BERT DUAL Dataset ========== #
class BERTDualDataset(Dataset):
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.res_max_len = max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
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
                (ids, sids, speaker), rids = self._length_limit(ids), self._length_limit_res(rids)
                self.data.append({
                    'ids': ids,
                    's_ids': sids,
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
                (ids, sids, speaker), rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    's_ids': sids,
                    'rids': rids,
                })    
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        speaker = 0
        sids = []
        for i in ids:
            sids.append(speaker)
            if i == self.sep:
                speaker = 1 if speaker == 0 else 0
        return ids, sids, speaker
    
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
            s_ids = torch.LongTensor(bundle['s_ids'])
            return ids, rids, s_ids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            s_ids = torch.LongTensor(bundle['s_ids'])
            return ids, rids, s_ids, bundle['label']

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
            s_ids = [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            s_ids = pad_sequence(s_ids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask, s_ids = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda(), s_ids.cuda()
            return ids, rids, ids_mask, rids_mask, s_ids
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, s_ids, label = batch[0], batch[1], batch[2], batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, s_ids, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), s_ids.cuda(), label.cuda()
            return ids, rids, rids_mask, s_ids, label


# ========== BERT FT Multi Dataset ========== # 
class BERTFTMultiDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_multi_ft.pt'
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
        return ids, tids, mask, label


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

# ========== BERT Generation ========== #
class BERTGenDataset(Dataset):
    
    '''segment embedding, token embedding, position embedding (default), mask embedding'''
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_gen.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data(path, lang=lang)
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
    
    def __init__(self, path, lang='zh', mode='train', max_len=300, model='bert-base-chinese'):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_gen_ft.pt'
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


class UMSDataset(Dataset):

    def __init__(self, path, mode='train', max_len=512, model='bert-base-chinese'):
        self.ins, self.del_, self.srch = True, True, True
        self.mode = mode
        self.max_len = max_len
        self.pp_path = f'{os.path.splitext(path)[0]}_ums.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.vocab = BertTokenizer.from_pretrained(model)
        if lang != 'zh':
            # add special tokens for english corpus, __number__, __path__, __url__
            self.vocab.add_tokens(['__number__', '__path__', '__url__'])
        self.data = read_ums_data(path, self.vocab)
        self.vocab.add_tokens(["[EOT]"])
        
        if self.ins:
            self.vocab.add_tokens(["[INS]"])
        if self.del_:
            self.vocab.add_tokens(["[DEL]"])
        if self.srch:
            self.vocab.add_tokens(["[SRCH]"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
        self.utterances = utterances
        self.response = response
        self.label
        """
        curr_example = self.data[index]
        current_feature = dict()
        anno_sent, segment_ids, attention_mask, eot_pos = self._annotate_sentence(curr_example)
        current_feature["res_sel"] = dict()
        current_feature["res_sel"]["anno_sent"] = torch.tensor(anno_sent).long()
        current_feature["res_sel"]["segment_ids"] = torch.tensor(segment_ids).long()
        current_feature["res_sel"]["attention_mask"] = torch.tensor(attention_mask).long()
        current_feature["res_sel"]["eot_pos"] = torch.tensor(eot_pos).long()
        current_feature["res_sel"]["label"] = torch.tensor(curr_example.label).float()

        # when the response is the ground truth, append it to utterances.
        if int(curr_example.label) == 1:
            curr_example.utterances.append(curr_example.response)

        if len(curr_example.utterances) == 1 and self.split == "train":
            return self._single_turn_processing(current_feature)

        if self.hparams.do_sent_insertion and self.split == "train":
            anno_sent, segment_ids, attention_mask, ins_pos, target_idx = self._insertion_annotate_sentence(curr_example)
            current_feature["ins"] = dict()
            current_feature["ins"]["anno_sent"] = torch.tensor(anno_sent).long()
            current_feature["ins"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["ins"]["attention_mask"] = torch.tensor(attention_mask).long()
            current_feature["ins"]["ins_pos"] = torch.tensor(ins_pos).long()
            current_feature["ins"]["label"] = torch.tensor(target_idx).long()

        if self.hparams.do_sent_deletion and self.split == "train":
            while True:
                target_idx = random.sample(list(range(self.num_input_examples)), 1)[0]
                target_example = self.input_examples[target_idx]
                if target_idx != index and len(target_example.utterances) > 2:
                    break
            anno_sent, segment_ids, attention_mask, del_pos, target_idx = self._deletion_annotate_sentence(curr_example,
                                                                                                     target_example)
            current_feature["del"] = dict()
            current_feature["del"]["anno_sent"] = torch.tensor(anno_sent).long()
            current_feature["del"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["del"]["attention_mask"] = torch.tensor(attention_mask).long()
            current_feature["del"]["del_pos"] = torch.tensor(del_pos).long()
            current_feature["del"]["label"] = torch.tensor(target_idx).long()

        if self.hparams.do_sent_search and self.split == "train":
            anno_sent, segment_ids, attention_mask, srch_pos, target_idx = self._search_annotate_sentence(curr_example)
            current_feature["srch"] = dict()
            current_feature["srch"]["anno_sent"] = torch.tensor(anno_sent).long()
            current_feature["srch"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["srch"]["attention_mask"] = torch.tensor(attention_mask).long()
            current_feature["srch"]["srch_pos"] = torch.tensor(srch_pos).long()
            current_feature["srch"]["label"] = torch.tensor(target_idx).long()

        return current_feature

    def _single_turn_processing(self, featrue: dict):

        max_seq_len = self.hparams.max_sequence_len
        if self.hparams.do_sent_insertion:
            featrue["ins"] = dict()
            featrue["ins"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["ins_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["label"] = torch.tensor(-1).long()

        if self.hparams.do_sent_deletion:
            featrue["del"] = dict()
            featrue["del"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["del_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["label"] = torch.tensor(-1).long()

        if self.hparams.do_sent_search:
            featrue["srch"] = dict()
            featrue["srch"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["srch_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["label"] = torch.tensor(-1).long()

        return featrue

    def _search_annotate_sentence(self, example):
        max_utt_len = self.hparams.max_utt_len
        num_utterances = len(example.utterances)
        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            example.utterances = example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = len(example.utterances)

        utt_len = 3  # cls sep sep
        for utt_id, utt in enumerate(example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                example.utterances[utt_id] = utt[:int(self.hparams.max_sequence_len / 4)]
            utt_len += len(utt) + 2  # srch, eot
            if utt_len > self.hparams.max_sequence_len:
                example.utterances = example.utterances[:utt_id]
                num_utterances = len(example.utterances)
                break

        target = example.utterances.pop() + ["[EOT]"]
        num_utterances -= 1

        random_utt_idx = list(range(num_utterances))
        random.shuffle(random_utt_idx)

        dialog_context = []
        target_idx = 0
        target_left = 0
        for i, random_id in enumerate(random_utt_idx):
            if random_id == num_utterances - 1:
                target_idx = i
                target_left = len(dialog_context)
            dialog_context.extend(["[SRCH]"] + example.utterances[random_id] + ["[EOT]"])

        target_right = len(dialog_context) - target_left
        dialog_context, target, target_idx = self._insert_max_len_trim_seq(dialog_context, target, target_idx,
                                                                       (target_left, target_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        target += ["[SEP]"]
        segment_ids.extend([1] * len(target))  # same utterance
        attention_mask.extend([1] * len(target))
    
        dialog_target = dialog_context + target

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        srch_pos = []
        srch_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[SRCH]":
                srch_pos.append(1)
                srch_cnt += 1
            else:
                srch_pos.append(0)

        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, srch_pos, target_idx

    def _deletion_annotate_sentence(self, curr_example, target_example):
        max_utt_len = self.hparams.max_utt_len - 1
    
        target_sentence = random.sample(target_example.utterances, 1)[0]
    
        # TODO: current example
        # current example -> deletion is included
        num_utterances = len(curr_example.utterances)
        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            curr_example.utterances = curr_example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = max_utt_len

        for utt_i, utt in enumerate(curr_example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                curr_example.utterances[utt_i] = utt[:int(self.hparams.max_sequence_len / 4)]

        curr_dialog_context = []
        delete_idx = random.sample(list(range(num_utterances)), 1)[0]
    
        delete_left = 0
        for utt_i, utt in enumerate(curr_example.utterances):
            if utt_i == delete_idx:
                delete_left = len(curr_dialog_context)
                curr_dialog_context.extend(["[DEL]"] + target_sentence + ["[EOT]"])
                if len(curr_example.utterances) > max_utt_len:
                    curr_example.utterances.pop()  # remove the last utterance
            curr_dialog_context.extend(["[DEL]"] + utt + ["[EOT]"])

        delete_right = len(curr_dialog_context) - delete_left

        target_dialog_context = []
        dialog_context, target_context, target_idx = \
      self._delete_max_len_trim_seq(curr_dialog_context, target_dialog_context, delete_idx, (delete_left, delete_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)
    
        dialog_target = dialog_context

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        del_pos = []
        del_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[DEL]":
                del_pos.append(1)
                del_cnt += 1
            else:
                del_pos.append(0)

        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, del_pos, target_idx

    def _insertion_annotate_sentence(self, example):
        max_utt_len = self.hparams.max_utt_len
    
        num_utterances = len(example.utterances)
    
        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            example.utterances = example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = len(example.utterances)

        for utt_i, utt in enumerate(example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                example.utterances[utt_i] = utt[:int(self.hparams.max_sequence_len / 4)]

        target = []
        dialog_context = ["[INS]"]
        target_idx = random.sample(list(range(num_utterances)), 1)[0]

        target_left, target_right = 0, 0
        for utt_i, utt in enumerate(example.utterances):
            if target_idx == utt_i:
                target_left = len(dialog_context) - 1
                target = utt + ["[EOT]"]
                continue
            dialog_context.extend(utt + ["[EOT]"] + ["[INS]"])

        target_right = len(dialog_context) - target_left
        dialog_context, target, target_idx = self._insert_max_len_trim_seq(dialog_context, target, target_idx,
                                                                       (target_left, target_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        target += ["[SEP]"]
        segment_ids.extend([1] * len(target))  # same utterance
        attention_mask.extend([1] * len(target))

        dialog_target = dialog_context + target

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        ins_pos = []
        ins_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[INS]":
                ins_pos.append(1)
                ins_cnt += 1
            else:
                ins_pos.append(0)
        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, ins_pos, target_idx

    def _annotate_sentence(self, example):

        dialog_context = []
        if self.hparams.do_eot:
            for utt in example.utterances:
                dialog_context.extend(utt + ["[EOT]"])
        else:
            for utt in example.utterances:
                dialog_context.extend(utt)
        response = example.response + ["[EOT]"]
        dialog_context, response = self._max_len_trim_seq(dialog_context, response)
    
        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        response = response + ["[SEP]"]
        segment_ids.extend([1] * len(response))
        attention_mask.extend([1] * len(response))

        dialog_response = dialog_context + response

        while len(dialog_response) < self.hparams.max_sequence_len:
            dialog_response.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        eot_pos = []
        for tok_idx, tok in enumerate(dialog_response):
            if tok == "[EOT]":
                eot_pos.append(1)
            else:
                eot_pos.append(0)

        assert len(dialog_response) == len(segment_ids) == len(attention_mask)
        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_response)
        assert len(dialog_response) <= self.hparams.max_sequence_len

        return anno_sent, segment_ids, attention_mask, eot_pos

    def _delete_max_len_trim_seq(self, curr_dialog_context, target_dialog_context, target_idx, lengths):
        delete_left, delete_right = lengths

        while len(curr_dialog_context) + len(target_dialog_context) > self.hparams.max_sequence_len - 3:
            if len(curr_dialog_context) > len(target_dialog_context):
                if delete_left > delete_right:
                    if curr_dialog_context[0] in ["[DEL]"]:
                        target_idx -= 1
                    delete_left -= 1
                    curr_dialog_context.pop(0)  # from the left
                else:
                    delete_right -= 1
                    curr_dialog_context.pop()  # from the right
            else:
                target_dialog_context.pop(0)

        return curr_dialog_context, target_dialog_context, target_idx

    def _insert_max_len_trim_seq(self, dialog_context, target, target_idx, lengths):

        target_left, target_right = lengths
        # [CLS] [SEP] [EOT] [SEP]
        while len(dialog_context) + len(target) > self.hparams.max_sequence_len - 3:
            if len(dialog_context) > len(target):
                if target_left > target_right:
                    if dialog_context[0] in ["[INS]"]:
                        target_idx -= 1
                    target_left -= 1
                    dialog_context.pop(0)  # from the left
                else:
                    target_right -= 1
                    dialog_context.pop()  # from the right
            else:
                target.pop()

        return dialog_context, target, target_idx

    def _max_len_trim_seq(self, dialog_context, response):

        while len(dialog_context) + len(response) > self.hparams.max_sequence_len - 3:
            if len(dialog_context) > len(response):
                dialog_context.pop(0)  # from the front
            else:
                response.pop()

        return dialog_context, response


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
        data = read_text_data_sa(path, lang=lang)
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
        'ums': UMSDataset,
        'sa-bert': SABERTFTDataset,
        'bert-gen': BERTGenDataset,
        'bert-ft': BERTFTDataset,
        'bert-ft-multi': BERTFTMultiDataset,
        'bert-gen-ft': BERTGenFTDataset,
        'dual-bert': BERTDualDataset,
        'dual-bert-mlm': BERTDualMLMDataset,
        'dual-bert-cross': BERTDualDataset,
        'dual-bert-scm': BERTDualDataset,
        'dual-bert-fg': BERTDualDataset,
        'dual-bert-jsd': BERTDualDataset,
        'dual-bert-gen': BERTDualGenDataset,
        'dual-bert-adv': BERTDualDataset,
        'dual-bert-poly': BERTDualDataset,
        'dual-bert-cl': BERTDualCLDataset,
        'dual-bert-vae': BERTDualDataset,
        'dual-bert-vae2': BERTDualDataset,
        'dual-bert-one2many': BERTDualOne2ManyDataset,
        'dual-bert-hierarchical': BERTDualHierarchicalDataset,
        'dual-bert-hierarchical-trs': BERTDualHierarchicalDataset,
        'dual-gru-hierarchical-trs': GRUDualHierarchicalDataset,
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

    if args['model'] == 'bert-ft-multi':
        path = f'data/{args["dataset"]}/{mode}_dup.txt'
    else:
        path = f'data/{args["dataset"]}/{mode}.txt'
    if mode == 'train':
        if args['model'] in ['dual-bert-one2many']:
            data = DATASET_MAP[args['model']](path, lang=args['lang'], mode=mode, max_len=args['max_len'], model=args['pretrained_model'], head=args['head_num'], res_max_len=args['res_max_len'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        elif args['model'] in ['dual-gru-hierarchical-trs']:
            vocab_path = f'data/{args["dataset"]}/word2vec.pt'
            data = DATASET_MAP[args['model']](path, vocab_path=vocab_path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        else:
            data = DATASET_MAP[args['model']](path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=dist.get_world_size(),
                rank=args['local_rank'],
            )
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
        sampler = train_sampler
    elif mode == 'inference':
        # path = f'data/{args["dataset"]}/train.txt'
        # data = BERTDualInferenceContextResponseDataset(path, lang=args['lang'], mode=mode, max_len=args['max_len'], model=args['pretrained_model'])
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     data,
        #     num_replicas=dist.get_world_size(),
        #     rank=args['local_rank'],
        # )
        # iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
        # sampler = None

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
        if args['model'] in ['dual-gru-hierarchical-trs']:
            vocab_path = f'data/{args["dataset"]}/word2vec.pt'
            data = DATASET_MAP[args['model']](path, vocab_path=vocab_path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        else:
            data = DATASET_MAP[args['model']](path, mode=mode, lang=args['lang'], max_len=args['max_len'], model=args['pretrained_model'])
            iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    if mode == 'inference':
        if not os.path.exists(data_ctx.pp_path):
            data_ctx.save()
        if not os.path.exists(data_res.pp_path):
            data_res.save()
    else:
        if not os.path.exists(data.pp_path):
            data.save()
    return data, iter_, sampler
