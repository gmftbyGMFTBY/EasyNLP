from header import *
from .utils import *


class BERTDualDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        self.pp_path = f'{os.path.splitext(path)[0]}_dual_{args["tokenizer"]}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_dual_bert(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
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
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
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
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda()
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask
            }
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
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label
            }


class BERTDualHierarchicalDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args

        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hier_{args["tokenizer"]}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                context = utterances[:-1]
                response = utterances[-1]
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
            data = read_text_data_dual_bert(path, lang=self.args['lang'])
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
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
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
        if self.args['mode'] == 'train':
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
            return {
                'cids': cids, 
                'rids': rids, 
                'cids_turn_length': cids_turn_length, 
                'cids_mask': cids_mask, 
                'rids_mask': rids_mask, 
                'recover_mapping': recover_mapping
            }
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
            return {
                'cids': cids, 
                'rids': rids, 
                'cids_turn_length': cids_turn_length, 
                'cids_mask': cids_mask, 
                'rids_mask': rids_mask, 
                'label': label
            }

class BERTDualWithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gray_{args["tokenizer"]}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data, responses = read_json_data(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                item = self.vocab.batch_encode_plus([context, response] + candidates)
                ids, rids = item['input_ids'][0], item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit_res(i) for i in rids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                item = self.vocab.batch_encode_plus([context, response] + candidates)
                ids = item['input_ids'][0]
                rids = item['input_ids'][1:]
                ids, rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [1] + [0] * 9,
                    'ids': ids,
                    'rids': rids,
                })    
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
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
        if self.args['mode'] == 'train':
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
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask
            }
        else:
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label
            }
