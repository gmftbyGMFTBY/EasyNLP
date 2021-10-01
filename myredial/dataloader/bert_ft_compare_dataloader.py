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
        self.topk = args['gray_cand_num']
        self.num_labels = args['num_labels']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_plus_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.responses = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.split(path)[0]}/train_bm25_gray.txt'
            data = read_bm25_hard_negative(path)
            responses, response_overlap = [], set()
            for item in tqdm(data):
                context, response, candidates = item['q'], item['r'], item['nr']
                ids = self.vocab.batch_encode_plus(context + [response], add_special_tokens=False)['input_ids']
                cids = []
                sids, cache = [], 0
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                    sids.extend([cache] * (len(u) + 1))
                    cache = 1 if cache == 0 else 0
                sids.pop()
                cids.pop()

                if len(cids) == 0:
                    # the empty sequence raise exception
                    continue

                rids = ids[-1]
                responses.append(rids)
                if response not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(response)
                self.data.append({
                    'context': cids,
                    'sids': sids,
                    'response': rids,
                    'candidates': candidates,
                })
            self.responses = responses
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                # too many validation samples, just sample 1000
                data = data[:10000]
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

    def _packup(self, cids, rids1, rids2, sids=None):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        sids_ = deepcopy(sids)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
            sids=sids_,
        )
        other_speaker = 0 if sids[-1] == 1 else 1
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        sids_ = [sids_[0]] + sids_ + [sids_[-1]] + [other_speaker] * (len(rids1_) + len(rids2_) + 2)
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [0] * (len(rids2_) + 1)
        assert len(sids_) == len(ids)
        return ids, tids, sids_

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['context'], bundle['response']
            speaker_ids = bundle['sids']

            if self.args['no_hard_negative']:
                hrids = random.sample(self.responses, self.topk)
            else:
                if self.topk > len(bundle['candidates']):
                    candidates = bundle['candidates']
                    if candidates:
                        hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
                    else:
                        hrids = []
                    hrids += random.sample(self.responses, self.topk - len(candidates))
                else:
                    candidates = random.sample(bundle['candidates'], self.topk)
                    hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            erids = random.sample(self.responses, self.topk)
            ids, sids, tids, label = [], [], [], []

            # label 0/1: positive vs. easy negative
            for e in erids:
                if random.random() > 0.5:
                    ids_, tids_, sids_ = self._packup(cids, rids, e, sids=speaker_ids)
                    l = 1
                else:
                    ids_, tids_, sids_ = self._packup(cids, e, rids, sids=speaker_ids)
                    l = 0
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                label.append(l)
            # label 0/1: positive vs. hard negatives
            for h in hrids:
                if random.random() > 0.5:
                    ids_, tids_, sids_ = self._packup(cids, rids, h, sids=speaker_ids)
                    l = 1
                else:
                    ids_, tids_, sids_ = self._packup(cids, h, rids, sids=speaker_ids)
                    l = 0
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                label.append(l)
            # whole samples
            ids = [torch.LongTensor(i) for i in ids]
            sids = [torch.LongTensor(i) for i in sids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, sids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                sids.extend(b[1])
                tids.extend(b[2])
                label.extend(b[3])
            label = torch.LongTensor(label)
            return {
                'ids': ids, 
                'sids': sids,
                'tids': tids, 
                'label': label
            }
        else:
            # test or valid set
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
