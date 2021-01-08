from header import *

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
    
# =========== load dataset ========== #
def load_dataset(args):
    DATASET_MAP = {
        'bert-gen': BERTGenDataset,
        'bert-ft': BERTFTDataset,
        'bert-gen-ft': BERTGenFTDataset,
    }
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] == 'train':
        data = DATASET_MAP[args['model']](path, mode=args['mode'], max_len=args['max_len'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
    else:
        data = DATASET_MAP[args['model']](path, mode=args['mode'], max_len=args['max_len'])
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    if not os.path.exists(data.pp_path):
        data.save()
    return data, iter_


if __name__ == "__main__":
    # ========== BERTFTDataset ========== #
    # train_data = BERTFTDataset('data/ecommerce/train.txt', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTFTDataset ========== #
    # ========== BERTGenDataset ========== #
    # train_data = BERTGenDataset('data/ecommerce/train.txt', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTGenDataset ========== #
    # ========== BERTGenFTDataset ========== #
    train_data = BERTGenFTDataset('data/ecommerce/train.txt', mode='train')
    train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=train_data.collate)
    # ========== BERTGenFTDataset ========== #
    train_data.save()
    for batch in tqdm(train_iter):
        ipdb.set_trace()
