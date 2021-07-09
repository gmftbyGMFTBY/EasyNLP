from header import *
from .utils import *


class BERTFTCompPlusDataset(Dataset):

    '''i vs. j
    0: fail; 1: tail; 2: win'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.k = args['k']
        self.data = []

        if self.args['mode'] == 'train':
            # read the train_gray.txt
            path = f'{os.path.splitext(path)[0]}_gray.txt'
            data = read_text_data_utterances_compare(path, lang=self.args['lang'])
            responses = [i[1] for i in data]
            self.responses = list(set(responses))
            for items in tqdm(data):
                context = ' [SEP] '.join(items[0])
                self.data.append({
                    'context': context,
                    'response': items[1],
                    'hard_negative_samples': items[2],
                    'super_hard_negative_samples': items[3],
                })
        else:
            # copy from dual-bert dataloader
            data = read_text_data_dual_bert(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[2] for b in batch]
                context = batch[0][1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
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

    def _encode_batch(self, texts, ctx=False):
        ids = self.vocab.batch_encode_plus(texts)['input_ids']
        if ctx:
            ids = [self._length_limit(i) for i in ids]
        else:
            ids = [self._length_limit_res(i) for i in ids]
        return ids

    def _packup(self, cids, rids1, rids2):
        ids = cids + rids1 + rids2
        tids = [0] * len(cids) + [1] * len(rids1) + [0] * len(rids2)
        return ids, tids

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            context, response = bundle['context'], bundle['response']
            super_hard_negative_samples = random.sample(bundle['super_hard_negative_samples'], self.k)
            hard_negative_samples = random.sample(bundle['hard_negative_samples'], self.k)
            easy_negative_samples = random.sample(self.responses, self.k)

            cids = self._encode_batch([context], ctx=True)[0]
            rids = self._encode_batch([response])[0]
            sh_rids = self._encode_batch(super_hard_negative_samples)
            e_rids = self._encode_batch(easy_negative_samples)
            h_rids = self._encode_batch(hard_negative_samples)

            ids, tids, label = [], [], []
            # label 0/2: positive vs. easy negative
            for e in e_rids:
                if random.random() > 0.5:
                    ids_, tids_ = self._packup(cids, rids, e)
                    l = 2
                else:
                    ids_, tids_ = self._packup(cids, e, rids)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # label 0/2: positive vs. hard negatives
            for h in h_rids:
                if random.random() > 0.5:
                    ids_, tids_ = self._packup(cids, rids, h)
                    l = 2
                else:
                    ids_, tids_ = self._packup(cids, h, rids)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # label 0/2: super hard vs. hard
            for _ in range(self.k):
                sh = random.choice(sh_rids)
                h = random.choice(h_rids)
                if random.random() > 0.5:
                    ids_, tids_ = self._packup(cids, sh, h)
                    l = 2
                else:
                    ids_, tids_ = self._packup(cids, h, sh)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # label 1: positive vs. super hard negatives
            for sh in sh_rids:
                if random.random() > 0.5:
                    ids_, tids_ = self._packup(cids, rids, sh)
                else:
                    ids_, tids_ = self._packup(cids, sh, rids)
                ids.append(ids_)
                tids.append(tids_)
                label.append(1)
            # label 1: self comparison
            for _ in range(self.k):
                ratio = random.random()
                if 0.9 < ratio <= 1:
                    r1, r2 = rids, rids
                elif 0.6 < ratio <= 0.9:
                    r1, r2 = random.sample(h_rids, 2)
                elif 0.3 < ratio <= 0.6:
                    r1, r2 = random.sample(e_rids, 2)
                else:
                    r1, r2 = random.sample(sh_rids, 2)
                ids_, tids_ = self._packup(cids, r1, r2)
                ids.append(ids_)
                tids.append(tids_)
                label.append(1)
            # 5*k samples
            ids = [torch.LongTensor(i) for i in ids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

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
            label = torch.LongTensor(label)
            return {
                'ids': ids, 
                'tids': tids, 
                'label': label
            }
        elif self.args['mode'] == 'test':
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }


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
            responses = []
            for items in tqdm(data):
                context = ' [SEP] '.join(items[0])
                self.data.append({
                    'context': context,
                    'response': items[1],
                    'hard_negative_samples': items[2],
                    'super_hard_negative_samples': items[3],
                })
                responses.append(items[1])
            self.responses = list(set(responses))
        else:
            # copy from dual-bert dataloader
            data = read_text_data_dual_bert(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[2] for b in batch]
                context = batch[0][1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
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

        if self.args['mode'] == 'train':
            context, response = bundle['context'], bundle['response']
            hard_negative_size = self.inner_bsz // 2
            super_hard_negative_size = self.inner_bsz - hard_negative_size 
            easy_negative_size = hard_negative_size 

            hard_negative_samples = random.sample(bundle['hard_negative_samples'], hard_negative_size)
            super_hard_negative_samples = random.sample(bundle['super_hard_negative_samples'], super_hard_negative_size)
            easy_negative_samples = random.sample(self.responses, easy_negative_size)

            cids, rids = self.vocab.batch_encode_plus([context, response])['input_ids']
            cids = self._length_limit(cids)
            rids = self._length_limit_res(rids)

            hrids = self.vocab.batch_encode_plus(hard_negative_samples)['input_ids']
            hrids = [self._length_limit_res(i) for i in hrids]
            shrids = self.vocab.batch_encode_plus(super_hard_negative_samples)['input_ids']
            shrids = [self._length_limit_res(i) for i in shrids]
            erids = self.vocab.batch_encode_plus(easy_negative_samples)['input_ids']
            erids = [self._length_limit_res(i) for i in erids]
            
            ids, tids, label = [], [], []
            # hard negative samples
            for h in hrids:
                if random.random() > 0.5:
                    ids_ = cids + rids + h
                    tids_ = [0] * len(cids) + [1] * len(rids) + [0] * len(h)
                    l = 1
                else:
                    ids_ = cids + h + rids
                    tids_ = [0] * len(cids) + [1] * len(h) + [0] * len(rids)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # easy negative samples
            for e in erids:
                if random.random() > 0.5:
                    ids_ = cids + rids + e
                    tids_ = [0] * len(cids) + [1] * len(rids) + [0] * len(e)
                    l = 1
                else:
                    ids_ = cids + e + rids
                    tids_ = [0] * len(cids) + [1] * len(e) + [0] * len(rids)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # super hard vs. hard only for training
            he_ids, he_tids, he_label = [], [], []
            for sh in shrids:
                for h in hrids:
                    if random.random() > 0.5:
                        ids_ = cids + sh + h
                        tids_ = [0] * len(cids) + [1] * len(sh) + [0] * len(h)
                        l = 1
                    else:
                        ids_ = cids + h + sh
                        tids_ = [0] * len(cids) + [1] * len(h) + [0] * len(sh)
                        l = 0
                    he_ids.append(ids_)
                    he_tids.append(tids_)
                    he_label.append(l)
            # random_idx = random.sample(range(len(he_ids)), len(erids))
            # he_ids = [he_ids[i] for i in random_idx]
            # he_tids = [he_tids[i] for i in random_idx]
            # he_label = [he_label[i] for i in random_idx]
            ids.extend(he_ids)
            tids.extend(he_tids)
            label.extend(he_label)

            ids = [torch.LongTensor(i) for i in ids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

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
        elif self.args['mode'] == 'test':
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }

class BERTFTCompEvaluationDataset(Dataset):
    
    '''Compare the evaluation results of the generated responses from two systems'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        # 22335: bm25+BERT-FP; 22336: dual-bert
        ports = args['file_tags'].split(',')
        path1 = f'{os.path.split(path)[0]}/test_api_pipeline_{ports[0]}_log.txt'
        path2 = f'{os.path.split(path)[0]}/test_api_pipeline_{ports[1]}_log.txt'
        print(f'[!] load file from:\n {path1}\n {path2}')
        
        data1 = read_text_data_from_log_file(path1, lang=args['lang'])
        data2 = read_text_data_from_log_file(path2, lang=args['lang'])

        self.data = []
        for (ctx1, res1), (ctx2, res2) in zip(data1, data2):
            assert ctx1 == ctx2
            self.data.append({
                'context': ctx1,
                'responses': [res1, res2]
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
        return self.data[i]

    def collate(self, batch):
        assert len(batch) == 1
        return batch[0]
