from header import *
from .utils import *
from .util_func import *
from .randomaccess import *
from config import *
from model import *


class BERTFTBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.pad_token_id
        self.sep = self.vocab.sep_token_id
        self.cls = self.vocab.cls_token_id
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        root_path = args['data_root_path']

        if self.args['mode'] == 'train':
            self.file_lists = [f'{root_path}/train_{i}.txt' for i in range(8)]
            self.current_file_index = 0
            self.current_file_handler = None
            self.cache = []
            self.buffer_size = args['buffer_size']

            # reset the random seed for each worker
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
        else:
            path = f'{root_path}/test.txt'
            data = read_text_data_utterances(path, lang=self.args['lang'])
            self.data = []
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids = []
                    for u in item[:-1]:
                        cids.extend(u + [self.eos])
                    cids.pop()
                    rids = item[-1]
                    truncate_pair(cids, rids, self.args['max_len'])
                    ids_ = [self.cls] + cids + [self.sep] + rids + [self.sep]
                    tids_ = [0] * (len(cids) + 2) + [1] * (len(rids) + 1)
                    ids.append(ids_)
                    tids.append(tids_)
                    responses.append(utterances[-1])
                context = ' [SEP] '.join(utterances[:-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                    'context': context,
                    'responses': responses
                })    
            self.size = len(self.data)
                
    def __len__(self):
        if self.args['mode'] == 'train':
            return 208779677
        else:
            return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) <= 10000:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                if len(self.cache) == 0:
                    # curretn file runs over, move to next file
                    self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)

            ids, tids, label = [], [], []

            # negative random sample
            random_line = random.choice(self.cache)
            random_line = json.loads(random_line)['data']
            random_utterance = random.choice(random_line)
            en = self.vocab.encode(random_utterance, add_special_tokens=False)

            line = self.cache.pop()
            line = json.loads(line)['data']
            items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
            cids = []
            for s in items[:-1]:
                cids.extend(s + [self.eos])
            cids.pop()
            rids = items[-1]

            cids_ = deepcopy(cids)
            truncate_pair(cids_, rids, self.args['max_len'])
            ids.append(torch.LongTensor([self.cls] + cids_ + [self.sep] + rids + [self.sep]))
            tids.append(torch.LongTensor([0] * (len(cids_) + 2) + [1] * (len(rids) + 1)))
            label.append(1)
            
            cids_ = deepcopy(cids)
            truncate_pair(cids_, en, self.args['max_len'])
            ids.append(torch.LongTensor([self.cls] + cids_ + [self.sep] + en + [self.sep]))
            tids.append(torch.LongTensor([0] * (len(cids_) + 2) + [1] * (len(en) + 1)))
            label.append(0)
            return ids, tids, label
        else:
            bundle = self.data[i]
            ids = torch.LongTensor(bundle['ids'])
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, rids, bundle['label'], bundle['context'], bundle['responses']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [], [], []
            for a, b, c in batch:
                ids.extend(a)
                tids.extend(b)
                label.extend(c)
            random_index = list(range(len(label)))
            random.shuffle(random_index)
            ids = [ids[i] for i in random_index]
            tids = [tids[i] for i in random_index]
            label = [label[i] for i in random_index]
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
        mask = generate_mask(ids)
        label = torch.LongTensor(label)
        ids, tids, mask, label = to_cuda(ids, tids, mask, label)
        if self.args['mode'] == 'train':
            return {
                'ids': ids, 
                'tids': tids, 
                'ids_mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'ids_mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

class DRBERTDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.pad_token_id
        self.sep = self.vocab.sep_token_id
        self.cls = self.vocab.cls_token_id
        root_path = args['data_root_path']

        if self.args['mode'] == 'train':
            self.file_lists = [f'{root_path}/train_{i}.txt' for i in range(8)]
            random.shuffle(self.file_lists)
            self.current_file_index = 0
            self.current_file_handler = None
            self.cache = []
            self.buffer_size = args['buffer_size']

            # reset the random seed for each worker
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
        else:
            path = f'{root_path}/test.txt'
            data = read_text_data_utterances(path, lang=self.args['lang'])
            self.data = []
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
            self.size = len(self.data)
                
    def __len__(self):
        if self.args['mode'] == 'train':
            return 208779677
        else:
            return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) == 0:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                if len(self.cache) == 0:
                    # curretn file runs over, move to next file
                    self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)
            line = self.cache.pop()
            line = json.loads(line)['data']
            items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
            ids = []
            for s in items[:-1]:
                ids.extend(s + [self.sep])
            ids.pop()
            ids = [self.cls] + ids[-self.args['max_len']:] + [self.sep]
            rids = [self.cls] + items[-1][:self.args['res_max_len']] + [self.sep]
            return torch.LongTensor(ids), torch.LongTensor(rids)
        else:
            bundle = self.data[i]
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence([i for i, j in batch], batch_first=True, padding_value=self.pad)
            rids = pad_sequence([j for i, j in batch], batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            rids_mask = generate_mask(rids, pad_token_idx=self.pad)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {'ids': ids, 'rids': rids, 'ids_mask': ids_mask, 'rids_mask': rids_mask}
        else:
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
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


class BERTCompBigDataset(Dataset):

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
        root_path = args['data_root_path']

        self.data = []
        if self.args['mode'] == 'train':
            self.file_lists = [f'{root_path}/train_{i}.txt' for i in range(8)]
            random.shuffle(self.file_lists)
            self.current_file_index = 0
            self.current_file_handler = None
            self.cache = []
            self.buffer_size = args['buffer_size']
            # reset the random seed for each worker
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
        else:
            path = f'{root_path}/test.txt'
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
        if self.args['mode'] == 'train':
            return 208779677
        else:
            return len(self.data)

    def _packup(self, cids, rids1, rids2):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
        )
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        cpids = [0] * (2 + len(cids_)) + [1] * (len(rids1_) + 1) + [2] * (len(rids2_) + 1)
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [1] * (len(rids2_) + 1)
        assert len(cpids) == len(ids) == len(tids)
        return ids, tids, cpids

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) == 0:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                if len(self.cache) == 0:
                    # curretn file runs over, move to next file
                    self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                    self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)
            line = self.cache.pop()
            line = json.loads(line)['data']
            items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
            cids = []
            rids = items[-1]
            for s in items[:-1]:
                cids.extend(s + [self.eos])
            cids.pop()

            if len(self.cache) <= self.args['random_sample_pool_size']:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                if len(self.cache) == 0:
                    # curretn file runs over, move to next file
                    self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                    self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)

            ids, tids, cpids, label = [], [], [], []
            # label 0/1: positive vs. easy negative
            for _ in range(self.topk):
                e = random.choice(
                    json.loads(random.choice(self.cache))['data']       
                )
                e = self.vocab.encode(e, add_special_tokens=False)
                if random.random() > 0.5:
                    ids_, tids_, cpids_ = self._packup(cids, rids, e)
                    l = 1
                else:
                    ids_, tids_, cpids_ = self._packup(cids, e, rids)
                    l = 0
                ids.append(ids_)
                tids.append(tids_)
                cpids.append(cpids_)
                label.append(l)
            # whole samples
            ids = [torch.LongTensor(i) for i in ids]
            cpids = [torch.LongTensor(i) for i in cpids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids, cpids, label
        else:
            bundle = self.data[i]
            # random shuffle
            random_idx = list(range(len(bundle['label'])))
            random.shuffle(random_idx)
            bundle['responses'] = [bundle['responses'][i] for i in random_idx]
            bundle['label'] = [bundle['label'][i] for i in random_idx]
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, cpids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                cpids.extend(b[2])
                label.extend(b[3])
            label = torch.LongTensor(label)
            return {
                'ids': ids, 
                'tids': tids, 
                'cpids': cpids,
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


class BERTCompHNBigDataset(Dataset):

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
        root_path = args['data_root_path']

        self.data = []
        if self.args['mode'] == 'train':
            self.file_lists = [f'{root_path}/train_{i}.txt' for i in range(8)]
            random.shuffle(self.file_lists)
            self.current_file_index = 0
            self.current_file_handler = None
            self.cache = []
            self.buffer_size = args['buffer_size']
            # reset the random seed for each worker
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)

            # load the dr-bert retrieval
            dr_bert_args = deepcopy(args)
            dr_bert_args['model'] = 'dual-bert'
            dr_bert_args['mode'] = 'test'
            config = load_config(dr_bert_args)
            dr_bert_args.update(config)

            self.dr_bert_agent = load_model(dr_bert_args)
            pretrained_model_name = dr_bert_args['pretrained_model'].replace('/', '_')
            save_path = f'{dr_bert_args["root_dir"]}/ckpt/{dr_bert_args["dataset"]}/{dr_bert_args["model"]}/best_{pretrained_model_name}_{dr_bert_args["version"]}.pt'
            self.dr_bert_agent.load_model(save_path)
            print(f'[!] init the dr-bert agent over')

            # load the bert-ft retrieval
            bert_ft_args = deepcopy(args)
            bert_ft_args['model'] = 'bert-ft'
            bert_ft_args['mode'] = 'test'
            config = load_config(bert_ft_args)
            bert_ft_args.update(config)

            # load the searcher
            self.searcher = Searcher(
                'IVF65536_HNSW32,PQ8',
                768,
                nprobe=5,
            )
            self.searcher.load(
                f'{dr_bert_args["root_dir"]}/data/{dr_bert_args["dataset"]}/dual-bert_{pretrained_model_name}_faiss.ckpt',        
                f'{dr_bert_args["root_dir"]}/data/{dr_bert_args["dataset"]}/dual-bert_{pretrained_model_name}_corpus.ckpt',        
            )


            self.bert_ft_agent = load_model(bert_ft_args)
            pretrained_model_name = bert_ft_args['pretrained_model'].replace('/', '_')
            save_path = f'{bert_ft_args["root_dir"]}/ckpt/{bert_ft_args["dataset"]}/{bert_ft_args["model"]}/best_{pretrained_model_name}_{bert_ft_args["version"]}.pt'
            self.bert_ft_agent.load_model(save_path)
            print(f'[!] init the bert-ft agent over')
        else:
            path = f'{root_path}/test.txt'
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
        if self.args['mode'] == 'train':
            return 208779677
        else:
            return len(self.data)

    def _packup(self, cids, rids1, rids2):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
        )
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        cpids = [0] * (2 + len(cids_)) + [1] * (len(rids1_) + 1) + [2] * (len(rids2_) + 1)
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [1] * (len(rids2_) + 1)
        assert len(cpids) == len(ids) == len(tids)
        return ids, tids, cpids

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) == 0:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                if len(self.cache) == 0:
                    # curretn file runs over, move to next file
                    self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                    print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                    self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)
            line = self.cache.pop()
            line = json.loads(line)['data']
            return line
        else:
            bundle = self.data[i]
            # random shuffle
            random_idx = list(range(len(bundle['label'])))
            random.shuffle(random_idx)
            bundle['responses'] = [bundle['responses'][i] for i in random_idx]
            bundle['label'] = [bundle['label'][i] for i in random_idx]
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            # retrieve
            context_list = [i[:-1] for i in batch]
            embd, _ = self.dr_bert_agent.inference_context_one_batch(context_list)
            hard_negative_samples = self.searcher._search(embd, topk=self.args['recall_pool_size'])
            # re-duplicate
            hard_negative = []
            for idx, item in enumerate(hard_negative_samples):
                gt = batch[idx][-1]
                p, pool = [], set()
                for u in item:
                    if u != gt and u not in pool:
                        pool.add(u)
                        p.append(u)
                hard_negative.append(p[:-1])
            
            # rerank
            rerank_batches = [{'context': context_list[i], 'candidates': hard_negative[i]} for i in range(len(batch))]
            hard_positive = self.bert_ft_agent.rerank_and_return(rerank_batches, rank_num=self.args['rank_num'], keep_num=self.args['gray_cand_num'], score_threshold=self.args['rank_score_threshold'], score_threshold_positive=self.args['rank_score_threshold_positive'])
            hard_negative = [random.sample(item[-self.args['rank_num']:], self.args['gray_cand_num']) for item in hard_negative]
            hard_positive = [[i for i, j in item] for item in hard_positive]

            easy_negative = self.sample_easy_negative(len(batch), self.args['gray_cand_num'])
            ids, tids, cpids, label = [], [], [], []
            for item, hn, en, hp in zip(batch, hard_negative, easy_negative, hard_positive):

                # hn size range from 0 to gray_cand_num
                # hn = hn[:self.args['gray_cand_num']]
                if random.random() < 0.05:
                    # replace one of the easy negative with the context utterance
                    uttr = random.choice(item[:-1])
                    en[0] = uttr

                items = self.vocab.batch_encode_plus(item + hn + en + hp, add_special_tokens=False)['input_ids']
                context = items[:len(item)-1]
                rids = items[len(item)-1]
                hardn = items[len(item):len(item)+len(hn)]
                easyn = items[len(item)+len(hn):len(item)+len(hn)+len(en)]
                hardp = items[-len(hp):]
                cids = []
                for s in context:
                    cids.extend(s + [self.eos])
                cids.pop()

                for hn_ in hardn:
                    if random.random() > 0.5:
                        ids_, tids_, cpids_ = self._packup(cids, rids, hn_)
                        l = 1
                    else:
                        ids_, tids_, cpids_ = self._packup(cids, hn_, rids)
                        l = 0
                    ids.append(ids_)
                    tids.append(tids_)
                    cpids.append(cpids_)
                    label.append(l)
                for en_ in easyn:
                    if random.random() > 0.5:
                        ids_, tids_, cpids_ = self._packup(cids, rids, en_)
                        l = 1
                    else:
                        ids_, tids_, cpids_ = self._packup(cids, en_, rids)
                        l = 0
                    ids.append(ids_)
                    tids.append(tids_)
                    cpids.append(cpids_)
                    label.append(l)
                for hp_ in hardp:
                    if random.random() > 0.5:
                        ids_, tids_, cpids_ = self._packup(cids, rids, hp_)
                    else:
                        ids_, tids_, cpids_ = self._packup(cids, hp_, rids)
                    l = 0.5
                    ids.append(ids_)
                    tids.append(tids_)
                    cpids.append(cpids_)
                    label.append(l)

            # whole samples
            ids = [torch.LongTensor(i) for i in ids]
            cpids = [torch.LongTensor(i) for i in cpids]
            tids = [torch.LongTensor(i) for i in tids]
            label = torch.tensor(label)
            return {
                'ids': ids, 
                'tids': tids, 
                'cpids': cpids,
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

    def sample_easy_negative(self, num, size):
        if len(self.cache) <= self.args['random_sample_pool_size']:
            if self.current_file_handler is None:
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
            self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            random.shuffle(self.cache)
        rest = []
        for _ in range(num):
            p = []
            for _ in range(size):
                item = random.choice(self.cache)
                utterance = random.choice(json.loads(item)['data'])
                p.append(utterance)
            rest.append(p)
        return rest



