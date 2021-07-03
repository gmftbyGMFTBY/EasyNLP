from header import *
from .utils import *


class BERTFTCompDataset(Dataset):

    '''make sure the ./scripts/inference.py[mode=gray] has been runed to generate the train_gray.txt data'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        # test model use all the negative samples to compare with the postitive samples
        self.inner_bsz = args['inner_bsz'] if self.args['mode'] == 'train' else 10 
        self.data = []

        if self.args['mode'] == 'train':
            # read the train_gray.txt
            path = f'{os.path.splitext(path)[0]}_gray.txt'
            data = read_text_data_utterances_compare(path, lang=self.args['lang'])
        else:
            # read the test.txt
            data = read_text_data_utterances_compare_test(path, lang=self.args['lang'])
        for items in tqdm(data):
            context = ' [SEP] '.join(items[0])
            self.data.append({
                'context': context,
                'response': items[1],
                'hard_negative_samples': items[2],
                'easy_negative_samples': items[3],
            })
                
    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            # minus 2: cls and sep tokens
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids

    def _length_limit_res(self, rids):
        if len(rids) > self.args['res_max_len']:
            # ignore the cls token
            rids = rids[1:self.args['res_max_len']] + [self.sep]
        else:
            rids = rids[1:]
        return rids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        context, response = bundle['context'], bundle['response']
        hard_negative_size = self.inner_bsz // 2
        easy_negative_size = self.inner_bsz - hard_negative_size 

        if self.args['mode'] == 'train':
            hard_negative_samples = random.sample(bundle['hard_negative_samples'], hard_negative_size)
            easy_negative_samples = random.sample(bundle['easy_negative_samples'], easy_negative_size)
        else:
            hard_negative_samples = bundle['hard_negative_samples']
            easy_negative_samples = bundle['easy_negative_samples']

        cids, rids = self.vocab.batch_encode_plus([context, response])['input_ids']
        cids = self._length_limit(cids)
        rids = self._length_limit_res(rids)

        hrids = self.vocab.batch_encode_plus(hard_negative_samples)['input_ids']
        hrids = [self._length_limit_res(i) for i in hrids]
        erids = self.vocab.batch_encode_plus(easy_negative_samples)['input_ids']
        erids = [self._length_limit_res(i) for i in erids]
        
        ids, tids, label = [], [], []
        # hard negative samples
        for h in hrids:
            if random.random() > 0.5:
                ids_ = cids + rids + h
                tids_ = [0] * len(cids) + [1] * len(rids) + [2] * len(h)
                l = 1
            else:
                ids_ = cids + h + rids
                tids_ = [0] * len(cids) + [1] * len(h) + [2] * len(rids)
                l = 0
            ids.append(ids_)
            tids.append(tids_)
            label.append(l)
        # easy negative samples
        for e in erids:
            if random.random() > 0.5:
                ids_ = cids + rids + e
                tids_ = [0] * len(cids) + [1] * len(rids) + [2] * len(e)
                l = 1
            else:
                ids_ = cids + e + rids
                tids_ = [0] * len(cids) + [1] * len(e) + [2] * len(rids)
                l = 0
            ids.append(ids_)
            tids.append(tids_)
            label.append(l)

        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        return ids, tids, label

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


class BERTFTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
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
                item = self.vocab.batch_encode_plus([context, response])
                cids, rids = item['input_ids']
                ids, tids = self._length_limit(cids, rids)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                cids, rids = [], []
                responses = []
                for b in batch:
                    context = ' [SEP] '.join(b[1][:-1])
                    response = b[1][-1]
                    responses.append(response)
                    cids_, rids_ = self.vocab.batch_encode_plus([context, response])['input_ids']
                    cids.append(cids_)
                    rids.append(rids_)
                ids, tids = [], []
                for c, r in zip(cids, rids):
                    ids_, tids_ = self._length_limit(c, r)
                    ids.append(ids_)
                    tids.append(tids_)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                    'context': context,
                    'responses': responses,
                })    
                
    def _length_limit(self, cids, rids):
        # cids
        if len(cids) > self.args['max_len']:
            cids = [cids[0]] + cids[-(self.args['max_len']-1):]     # [CLS] ... [SEP]
        # rids: without [CLS] token
        if len(rids) > self.args['res_max_len']:
            rids = rids[1:self.args['res_max_len']] + [self.sep] 
        else:
            rids = rids[1:]

        ids = cids + rids
        tids = [0] * len(cids) + [1] * len(rids)
        return ids, tids
                
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

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
            context, responses = [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
                context.append(b[3])
                responses.extend(b[4])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, mask, label = ids.cuda(), tids.cuda(), mask.cuda(), label.cuda()
        if self.args['mode'] == 'train':
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

class BERTFTWithNegDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.data = []
        if self.args['mode'] == 'train':
            data, responses = read_text_data_with_neg_q_r_neg(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context)
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]

                ids = item['input_ids']
                tids = item['token_type_ids']
                ids = self._length_limit(ids)
                tids = self._length_limit(tids)
                self.data.append({
                    'label': [1] + [0] * 10, 
                    'text': [(context, res) for res in [response] + candidate]
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

    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            label = bundle['label']
            texts = bundle['text']
            item = self.vocab.batch_encode_plus(texts)
            ids = [torch.LongTensor(self._length_limit(i)) for i in item['input_ids']]
            tids = [torch.LongTensor(self._length_limit(i)) for i in item['token_type_ids']]
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
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
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
