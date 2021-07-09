from header import *
from .utils import *
from .util_func import *


class BERTFTCompDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.k = args['k']
        self.num_labels = args['num_labels']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_plus_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.responses = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.splitext(path)[0]}_gray.txt'
            data = read_text_data_utterances_compare(path, lang=self.args['lang'])
            responses = []
            response_overlap = set()
            for items in tqdm(data):
                ids = self.vocab.batch_encode_plus(items[0] + [items[1]], add_special_tokens=False)['input_ids']
                cids = []
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                cids.pop()
                rids = ids[-1]
                ids = self.vocab.batch_encode_plus(items[2] + items[3], add_special_tokens=False)['input_ids']
                hrids = ids[:len(items[2])]
                shrids = ids[len(items[2]):]
                responses.append(rids)
                if items[1] not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(items[1])

                self.data.append({
                    'context': cids,
                    'response': rids,
                    'hard_negative_samples': hrids,
                    'super_hard_negative_samples': shrids,
                })
            self.responses = responses
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[1][-1] for b in batch]
                context = batch[0][1][:-1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def _packup(self, cids, rids1, rids2):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len']
        )
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [0] * (len(rids2_) + 1)
        return ids, tids

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['context'], bundle['response']
            sh_rids = random.sample(bundle['super_hard_negative_samples'], self.k)
            h_rids = random.sample(bundle['hard_negative_samples'], self.k)
            e_rids = random.sample(self.responses, self.k)

            ids, tids, label = [], [], []
            # label 0/2: positive vs. easy negative
            for e in e_rids:
                if random.random() > 0.5:
                    ids_, tids_ = self._packup(cids, rids, e)
                    l = 1 if self.num_labels == 1 else 2
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
                    l = 1 if self.num_labels == 1 else 2
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
                    l = 1 if self.num_labels == 1 else 2
                else:
                    ids_, tids_ = self._packup(cids, h, sh)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                label.append(l)
            # label 1: self comparison
            if self.num_labels == 3:
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
            # 4*k samples
            ids = [torch.LongTensor(i) for i in ids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        data = torch.save((self.data, self.responses), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
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
                'context': [i.strip() for i in ctx1.split(' [SEP] ')],
                'responses': [res1, res2]
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):
        assert len(batch) == 1
        return batch[0]
