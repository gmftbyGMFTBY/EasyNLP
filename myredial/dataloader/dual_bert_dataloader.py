from header import *
from .utils import *
from .util_func import *


class BERTDualDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])

        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }


class BERTDualHierarchicalDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args

        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hier_{suffix}.pt'
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
            cids_mask = [generate_mask(item).cuda() for item in cids]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            rids, rids_mask = to_cuda(rids, rids_mask)
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
            rids_mask = generate_mask(rids)
            cids_mask = generate_mask(cids)
            label = torch.LongTensor(label)
            cids, rids, cids_mask, rids_mask, label = to_cuda(cids, rids, cids_mask, rids_mask, label)
            return {
                'ids': cids, 
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
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gray_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        if self.args['mode'] == 'train':
            # data, responses = read_text_data_with_neg_q_r_neg(path, lang=self.args['lang'])
            # data, responses = read_text_data_with_super_hard_q_r(path, lang=self.args['lang'])
            data, responses = read_text_data_with_neg_inner_session_neg(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                item = self.vocab.batch_encode_plus([context, response] + candidates)
                ids, rids = item['input_ids'][0], item['input_ids'][1:]
                ids, rids = length_limit(ids, self.args['max_len']), [length_limit_res(i, self.args['res_max_len'], sep=self.sep) for i in rids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            data = read_text_data_dual_bert(path, lang='zh')
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for item in batch:
                    item = self.vocab.batch_encode_plus([item[1], item[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                ids, rids = length_limit(ids, self.args['max_len']), [length_limit_res(rids_, self.args['res_max_len'], sep=self.sep) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids, 
                    'rids': rids,
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            pos_rids = bundle['rids'][0]
            # sample based on the gray_cand_num parameter
            rids = [pos_rids] + random.sample(bundle['rids'][1:], self.args['gray_cand_num'])
            rids = [torch.LongTensor(i) for i in rids]
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
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
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label
            }

class BERTDualCLDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dualCL_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                context_utterances = utterances[:-1]
                response = utterances[-1]
                context = ' [SEP] '.join(context_utterances)

                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids']
                ids, rids = length_limit(ids, self.args['max_len']), length_limit_res(rids, self.args['res_max_len'], sep=self.sep)
                self.data.append({
                    'ids': ids,
                    'context_utterances': context_utterances,
                    'rids': rids,
                })
        else:
            data = read_text_data_dual_bert(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for item_ in batch:
                    item = self.vocab.batch_encode_plus([item_[1], item_[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                    if item_[0] == 1:
                        gt_text.append(item_[2])
                ids, rids = length_limit(ids, self.args['max_len']), [length_limit_res(rids_, self.args['res_max_len'], sep=self.sep) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            # random
            context_utterances = bundle['context_utterances']
            length = len(context_utterances) - 1
            candidate_context_l = random.randint(0, length)
            candidate_context = context_utterances[-1-candidate_context_l:]
            candidate_context = ' [SEP] '.join(candidate_context)
            ids_cand = length_limit(self.vocab.encode(candidate_context), self.args['max_len'])
            ids_cand = torch.LongTensor(ids_cand)
            return ids, ids_cand, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, ids_cand, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_cand = pad_sequence(ids_cand, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids_cand_mask = generate_mask(ids_cand)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            ids_cand, ids_cand_mask = to_cuda(ids_cand, ids_cand_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_cand': ids_cand,
                'ids_cand_mask': ids_cand_mask,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualO2MDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
       
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gray_o2m_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            # data = read_text_data_with_super_hard_q_r(path, lang=self.args['lang'])
            # data = read_text_data_one2many_pesudo(path, lang=self.args['lang'])
            data = read_text_data_one2many(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                item = self.vocab.batch_encode_plus(context + [response] + candidates, add_special_tokens=False)['input_ids']
                cids = item[:len(context)]
                rids = item[len(context)]
                cand_rids = item[-len(candidates):]

                ids = [self.cls]
                for u in cids:
                    ids.extend(u + [self.eos])
                ids[-1] = self.sep
                ids = length_limit(ids, self.args['max_len'])
                rids = length_limit_res([self.cls] + rids + [self.sep], self.args['res_max_len'], sep=self.sep)
                cand_rids = [length_limit_res([self.cls] + i + [self.sep], self.args['res_max_len'], sep=self.sep) for i in cand_rids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'cand_rids': cand_rids,
                })
        else:
            data = read_text_data_utterances(path, lang='zh')
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    rids.append([self.cls] + item[-1] + [self.sep])
                ids = [self.cls]
                for u in item[:-1]:
                    ids.extend(u + [self.eos])
                ids[-1] = self.sep
                ids = length_limit(ids, self.args['max_len'])
                rids = [length_limit_res(i, self.args['res_max_len'], sep=self.sep) for i in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids, 
                    'rids': rids,
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            pos_rids = bundle['rids']
            rids = [pos_rids] + random.sample(bundle['cand_rids'], self.args['gray_cand_num'])
            rids = [torch.LongTensor(i) for i in rids]
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids = to_cuda(ids)[0]
            rids, rids_mask = [], []
            for i in range(0, self.args['gray_cand_num']+1):
                rids_ = [item[1][i] for item in batch]
                rids_ = pad_sequence(rids_, batch_first=True, padding_value=self.pad)
                rids_mask_ = generate_mask(rids_)
                rids_, rids_mask_ = to_cuda(rids_, rids_mask_)
                rids.append(rids_)
                rids_mask.append(rids_mask_)
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
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label
            }

            
class BERTDualFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])

        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                start_num = max(1, len(item) - 5) 
                for i in range(start_num, len(item)):
                    cids, rids = item[:i], item[i]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids = rids[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids = [self.cls] + rids + [self.sep]
                    self.data.append({
                        'ids': ids,
                        'rids': rids,
                        'ctext': ' [SEP] '.join(utterances[:i]),
                        'rtext': utterances[i],
                    })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualPseudoDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_pseudo_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None


        self.data = []
        if self.args['mode'] == 'train':
            pseudo_path = f'{os.path.splitext(path)[0]}_gray_unparallel.txt'
            # data = read_text_data_utterances_and_pesudo_pairs(path, pseudo_path, lang=self.args['lang'])
            data = read_text_data_utterances_and_full_and_pesudo_pairs(path, pseudo_path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullFakeCtxDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[CTX]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.ctx = self.vocab.convert_tokens_to_ids('[CTX]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_fullfakectx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.ext_data = torch.load(self.pp_path)
            elif self.args['mode'] == 'test':
                self.data = torch.load(self.pp_path)
            else:
                raise Exception(f'[!] Unknown mode: {self.args["mode"]}')

            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            self.ext_data = []
            ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            data, ext_data = read_text_data_utterances_full_fake_ctx(path, ext_path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
            for label, utterances in tqdm(ext_data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['res_max_len']-4):]    # [CLS] [CTX] ids [CTX] [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls, self.ctx] + ids + [self.ctx, self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.ext_data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(bundle['ids'])]
            rids = [torch.LongTensor(bundle['rids'])]
            ctext = [bundle['ctext']]
            rtext = [bundle['rtext']]
            # recall some fake data to train with the groundtruth pairs
            fake_num = self.args['fake_num']
            bundles = random.sample(self.ext_data, fake_num)
            for b in bundles:
                ids.append(torch.LongTensor(b['ids']))
                rids.append(torch.LongTensor(b['rids']))
                ctext.append(b['ctext'])
                rtext.append(b['rtext'])
            return ids, rids, ctext, rtext
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.ext_data), self.pp_path)
        elif self.args['mode'] == 'test':
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids, ctext, rtext = [], [], [], []
            for ids_, rids_, ctext_, rtext_ in batch:
                ids.extend(ids_)
                rids.extend(rids_)
                ctext.extend(ctext_)
                rtext.extend(rtext_)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text,
            }

            
class BERTDualExtraNoCtxDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_extra_noctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.ext_data = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None


        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
            # extended unsupervised utterances
            ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            ext_data = read_extended_douban_corpus(ext_path)
            self.ext_data = []
            inner_bsz = 256
            for idx in tqdm(range(0, len(ext_data), inner_bsz)):
                utterances = ext_data[idx:idx+inner_bsz]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                rids = [[self.cls] + i[:self.args['res_max_len']-2] + [self.sep] for i in item]
                self.ext_data.extend(rids)
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        if self.args['mode'] == 'train':
            torch.save((self.data, self.ext_data), self.pp_path)
        else:
            torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]

            # extended dataset
            ext_rids = random.sample(self.ext_data, len(batch)*self.args['ext_num'])
            ext_rids = [torch.LongTensor(i) for i in ext_rids]
            rids += ext_rids

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSemiDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_semi_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.ext_data = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
            # extended unsupervised utterances
            if args['dataset'] in ['restoration-200k', 'douban']:
                ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            else:
                ext_path = f'{args["root_dir"]}/data/{args["dataset"]}/ext_corpus.txt'
            ext_data = read_extended_douban_corpus(ext_path)
            self.ext_data = []
            inner_bsz = 256
            for idx in tqdm(range(0, len(ext_data), inner_bsz)):
                utterances = ext_data[idx:idx+inner_bsz]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                rids = [[self.cls] + i[:self.args['res_max_len']-2] + [self.sep] for i in item]
                self.ext_data.extend(rids)
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        if self.args['mode'] == 'train':
            torch.save((self.data, self.ext_data), self.pp_path)
        else:
            torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]

            # extended dataset
            ext_rids = random.sample(self.ext_data, len(batch)*self.args['ext_num'])
            ext_rids = [torch.LongTensor(i) for i in ext_rids]
            ext_rids = pad_sequence(ext_rids, batch_first=True, padding_value=self.pad)
            ext_rids_mask = generate_mask(ext_rids)

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            rids_mask = generate_mask(rids)
            ids, rids, ext_rids, ids_mask, rids_mask, ext_rids_mask = to_cuda(ids, rids, ext_rids, ids_mask, rids_mask, ext_rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ext_rids': ext_rids,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ext_rids_mask': ext_rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }
