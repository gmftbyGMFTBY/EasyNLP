from header import *
from randomaccess import RandomAccessReader
from .utils import *
from .util_func import *


class PostTrainDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.table = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        self.table = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            offset = len(self.data)
            self.data.extend(item)

            counter = 0
            l = []
            for utterance in item:
                l.append(len([i for i in utterance if i not in self.special_tokens]))
            # begin, end, max-length session
            for i in range(1, len(item)):
                if i < self.args['min_context_length']:
                    continue
                # check if the context and response are legal
                # if sum(l[:i+1]) > self.args['min_token_length'] and l[i] > 0:
                #     self.table.append((offset, offset+i, len(self.data)))

                if l[i] > 0:
                    self.table.append((offset, offset+i, len(self.data)))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        begin, end, max_l = self.table[i]
        session = self.data[begin:end+1]
        tokens = []
        for utterance in session[:-1]:
            tokens.extend(utterance + [self.eos])
        tokens.pop()

        ratio = random.random()
        if ratio > 0.75:
            # ground-truth
            response = session[-1]
            label = 2
        elif ratio > 0.5:
            # within session
            index = list(range(begin, max_l))
            index.remove(end)
            response = self.data[random.choice(index)]
            label = 1
        else:
            # random negative sample
            while True:
                rand_idx = random.randint(0, len(self.data)-1)
                if rand_idx != end:
                    break
            response = self.data[rand_idx]
            label = 0

        response_ = deepcopy(response)
        truncate_pair(tokens, response_, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response_ + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response_) + 1)
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, tids, mask_labels, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, mask_labels, labels = [], [], [], []
        for ids_, tids_, mask_labels_, labels_ in batch:
            ids.append(ids_)
            tids.append(tids_)
            mask_labels.append(mask_labels_)
            labels.append(labels_)
        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        labels = torch.LongTensor(labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, mask_labels, attn_mask, labels = to_cuda(ids, tids, mask_labels, attn_mask, labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }

class PostTrainMonoDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_mono_ext_{self.args["ext_read"]}_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # for restoration-200k
        # for douban, ecommerce, ubuntu, restoration-200k just on their own dataset
        data = read_text_data_utterances(path, lang=args['lang'])
        data = list(chain(*[utterances for l, utterances in data if l == 1]))
        # also add the extended nonparallel corpus
        # in re-rank exp, do not use it; in full-rank exp, use it.
        if self.args['dataset'] in ['restoration-200k'] and self.args['ext_read']:
            ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            data += read_extended_douban_corpus(ext_path)
        data = list(set(data))

        self.data = []
        for utterance in tqdm(data):
            item = self.vocab.encode(utterance, add_special_tokens=False)
            item = item[:self.args['max_len']-2]
            num_valid = len([i for i in item if i not in self.special_tokens])
            if num_valid < self.args['min_len']:
                continue
            self.data.append(item)
        print(f'[!] dataset size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tokens = self.data[i]
        ids = [self.cls] + tokens + [self.sep]
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, mask_labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.data)}')
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for ids_, mask_labels_ in batch:
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }

        
class PostTrainNoCLSDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_no_cls_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.table = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        self.table = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            offset = len(self.data)
            self.data.extend(item)

            counter = 0
            l = []
            for utterance in item:
                l.append(len([i for i in utterance if i not in self.special_tokens]))
            # begin, end, max-length session
            for i in range(1, len(item)):
                if i < self.args['min_context_length']:
                    continue
                # check if the context and response are legal
                if sum(l[:i+1]) > self.args['min_token_length'] and l[i] > 0:
                    self.table.append((offset, offset+i, len(self.data)))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        begin, end, max_l = self.table[i]
        session = self.data[begin:end+1]
        tokens = []
        for utterance in session[:-1]:
            tokens.extend(utterance + [self.sep])
        tokens.pop()
        response = session[-1]

        truncate_pair(tokens, response, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, tids, mask_labels

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, mask_labels = [], [], []
        for ids_, tids_, mask_labels_ in batch:
            ids.append(ids_)
            tids.append(tids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, mask_labels, attn_mask = to_cuda(ids, tids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }

        
class PostTrainMultiStrategiesDataset(Dataset):

    '''labels:
    (at least means the length of the context, not include the response)
    0: [SAFE] random negative response, at least 1 utterance
    1: [SAFE] within hard negative sample response, at least 1 utterance
    2: [SAFE] positive response, at least 1 utterance
    3: [NOT SAFE] random utterance insert into the dialog (random negative within multi-turn dialog, not the last turn), at least 1 utterances in multi-turn conversation session
    4: [SAFE] random shuffle the utterance, at least 1 utterances
    5: [NOT SAFE] random delete one utterance in dialog, at least 2 utterances, and cannot delete the right and left utterance, make sure the delete utterance will be count in the max_len sequences 
    6: [NOT SAFE] random replace the utterance, cannot replace the last one, at least 2 utterances

    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.retry_time = args['retry_time']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_multi_strategy_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.table = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        self.table = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            offset = len(self.data)

            counter = 0
            l, item_n = [], []
            for utterance in item:
                length = len([i for i in utterance if i not in self.special_tokens])
                if length > 0:
                    item_n.append(utterance)
                    l.append(length)
            
            self.data.extend(item_n)
            # (begin, end, max-length) for one session
            for i in range(1, len(item_n)):
                self.table.append((
                    offset, 
                    offset+i, 
                    len(self.data)
                ))

    def __len__(self):
        return len(self.table)
    
    def packup_tokens(self, session):
        tokens = []
        for utterance in session:
            tokens.extend(utterance + [self.eos])
        tokens.pop()
        return tokens

    def build_for_label_0(self, session, end):
        # random negative sample
        while True:
            rand_idx = random.randint(0, len(self.data)-1)
            if rand_idx != end:
                break
        response = self.data[rand_idx]
        # pack up the token ids
        tokens = self.packup_tokens(session[:-1])
        truncate_pair(tokens, response, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        return ids, tids

    def build_for_label_1(self, begin, end, max_l, session):
        # within session
        index = list(range(begin, max_l))
        index.remove(end)
        response = self.data[random.choice(index)]
        # pack up the token ids
        tokens = self.packup_tokens(session[:-1])
        truncate_pair(tokens, response, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        return ids, tids

    def build_for_label_2(self, session):
        # ground-truth
        tokens = self.packup_tokens(session[:-1])
        response = session[-1]
        truncate_pair(tokens, response, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        return ids, tids

    def build_for_label_3(self, begin, max_l, session_origin):
        # insert one random utterance into context
        for _ in range(self.retry_time):
            session = deepcopy(session_origin)
            # random response sample
            while True:
                rand_idx = random.randint(0, len(self.data)-1)
                if rand_idx not in set(range(begin, max_l)):
                    break
            random_response = self.data[rand_idx][:self.args['res_max_len']-2]
            
            idx = random.choice(range(len(session)))
            session_label = [[0] * len(u) for u in session]
            session[idx:idx] = [random_response]
            session_label[idx:idx] = [[1] * len(random_response)]
            ids, labels = [], []
            for u, l in zip(session[:-1], session_label[:-1]):
                ids.extend(u + [self.eos])
                labels.extend(l + [l[-1]])
            ids.pop()
            labels.pop()
            response = deepcopy(session[-1])
            truncate_pair_with_labels(ids, labels, response, self.args['max_len'])
            if sum(labels) > 0:
                ids_ = [self.cls] + ids + [self.sep] + response + [self.sep]
                tids = [0] * (len(ids) + 2) + [1] * (len(response) + 1)
                return ids_, tids
        # Fail to generate the data in label 3
        return None, None

    def build_for_label_4(self, session):
        # random shuffle
        random_idx = list(range(len(session)))
        while True:
            # multi shuffle
            for _ in range(self.args['shuffle_time']):
                random.shuffle(random_idx)
            if random_idx != list(range(len(session))):
                break
        session = [session[i] for i in random_idx]
        # ground-truth
        tokens = self.packup_tokens(session[:-1])
        response = session[-1]
        truncate_pair(tokens, response, self.args['max_len'])
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        return ids, tids

    def build_for_label_5(self, session_origin):
        # randomly delete one utterance, cannot delete the right or left utterance
        for _ in range(self.retry_time):
            session = deepcopy(session_origin)
            labels = [[idx] * len(u) for idx, u in enumerate(session)]
            idx = random.choice(range(1, len(session)-1))
            tokens = [u for i, u in enumerate(session) if i != idx]
            labels = [u for i, u in enumerate(labels) if i != idx]

            ids, labels_ = [], []
            for u, l in zip(tokens[:-1], labels[:-1]):
                ids.extend(u + [self.eos])
                labels_.extend(l + [l[-1]])
            ids.pop()
            labels_.pop()

            rids_labels = labels[-1]
            response = tokens[-1]
            truncate_pair_with_labels(ids, labels_, response, self.args['max_len'], rids_labels=rids_labels)
            # check
            labels_ += rids_labels
            flag = False
            for idx in range(1, len(labels_)):
                if labels_[idx] - labels_[idx-1] in [0, 1]:
                    pass
                else:
                    flag = True
                    break
            if flag:
                ids_ = [self.cls] + ids + [self.sep] + response + [self.sep]
                tids = [0] * (len(ids) + 2) + [1] * (len(response) + 1)
                return ids_, tids
        return None, None

    def build_for_label_6(self, begin, max_l, session_origin):
        # random replace one utterance, cannot replace the last one
        for _ in range(self.retry_time):
            session = deepcopy(session_origin)
            # random sample idx
            while True:
                rand_idx = random.randint(0, len(self.data)-1)
                if rand_idx not in set(range(begin, max_l)):
                    break
            random_response = self.data[rand_idx][:self.args['res_max_len']-2]

            idx = random.choice(range(len(session) - 1))
            labels = [[0] * len(u) for u in session]
            session[idx] = random_response
            labels[idx] = [1] * len(random_response)
            
            ids, labels_ = [], []
            for u, l in zip(session[:-1], labels[:-1]):
                ids.extend(u + [self.eos])
                labels_.extend(l + [l[-1]])
            ids.pop()
            labels_.pop()

            rids = deepcopy(session[-1])
            rids_labels = labels[-1]
            truncate_pair_with_labels(ids, labels_, rids, self.args['max_len'], rids_labels=rids_labels)
            if sum(labels_) > 0: 
                ids_ = [self.cls] + ids + [self.sep] + rids + [self.sep]
                tids = [0] * (len(ids) + 2) + [1] * (len(rids) + 1)
                return ids_, tids
        return None, None

    def __getitem__(self, i):
        begin, end, max_l = self.table[i]
        session = self.data[begin:end+1]
        # avoid the very long utterance
        session = [u[:self.args['res_max_len']-2] for u in session]

        while True:
            ratio = random.random()
            if len(session) >= 3:
                # 7 division
                if ratio >= 1 - 1/7:
                    ids, tids = self.build_for_label_0(session, end)
                    label = 0
                elif 1 - 2/7 <= ratio < 1 - 1/7:
                    ids, tids = self.build_for_label_1(begin, end, max_l, session)
                    label = 1
                elif 1 - 3/7 <= ratio < 1 - 2/7:
                    ids, tids = self.build_for_label_2(session)
                    label = 2
                elif 1 - 4/7 <= ratio < 1 - 3/7:
                    ids, tids = self.build_for_label_3(begin, max_l, session)
                    label = 3
                elif 1 - 5/7 <= ratio < 1 - 4/7:
                    ids, tids = self.build_for_label_4(session)
                    label = 4
                elif 1 - 6/7 <= ratio < 1 - 5/7:
                    ids, tids = self.build_for_label_5(session)
                    label = 5
                elif 1 - 7/7 <= ratio < 1 - 6/7:
                    ids, tids = self.build_for_label_6(begin, max_l, session)
                    label = 6
            else:
                # 5 division
                if ratio >= 1 - 1/5:
                    ids, tids = self.build_for_label_0(session, end)
                    label = 0
                elif 1 - 2/5 <= ratio < 1 - 1/5:
                    ids, tids = self.build_for_label_1(begin, end, max_l, session)
                    label = 1
                elif 1 - 3/5 <= ratio < 1 - 2/5:
                    ids, tids = self.build_for_label_2(session)
                    label = 2
                elif 1 - 4/5 <= ratio < 1 - 3/5:
                    ids, tids = self.build_for_label_3(begin, max_l, session)
                    label = 3
                elif 1 - 5/5 <= ratio < 1 - 4/5:
                    ids, tids = self.build_for_label_4(session)
                    label = 4
            if ids is not None and tids is not None:
                break
        
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, tids, mask_labels, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, mask_labels, tids, labels = [], [], [], []
        for ids_, tids_, mask_labels_, labels_ in batch:
            ids.append(ids_)
            tids.append(tids_)
            mask_labels.append(mask_labels_)
            labels.append(labels_)
        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        labels = torch.LongTensor(labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, mask_labels, attn_mask, labels = to_cuda(ids, tids, mask_labels, attn_mask, labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }

        
class PostTrainComparisonDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_compare_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.table = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        self.table = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            offset = len(self.data)
            self.data.extend(item)

            counter = 0
            l = []
            for utterance in item:
                l.append(len([i for i in utterance if i not in self.special_tokens]))
            # begin, end, max-length session
            for i in range(1, len(item)):
                if i < self.args['min_context_length']:
                    continue
                # check if the context and response are legal
                # if sum(l[:i+1]) > self.args['min_token_length'] and l[i] > 0:
                #     self.table.append((offset, offset+i, len(self.data)))

                if l[i] > 0:
                    self.table.append((offset, offset+i, len(self.data)))

    def __len__(self):
        return len(self.table)
    
    def _packup(self, cids, rids1, rids2, sids):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        sids_ = deepcopy(sids)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
            sids=sids_,
        )
        next_speaker = 1 if sids_[-1] == 0 else 0
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        cpids = [0] * (2 + len(sids_)) + [1] * (len(rids1_) + 1) + [2] * (len(rids2_) + 1)
        sids_ = [sids_[0]] + sids_ + [sids[-1]] + [next_speaker] * (len(rids1_) + len(rids2_) + 2)
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [1] * (len(rids2_) + 1)
        assert len(sids_) == len(ids)
        assert len(cpids) == len(ids)
        return ids, tids, sids_, cpids

    def __getitem__(self, i):
        begin, end, max_l = self.table[i]
        session = self.data[begin:end+1]
        cids, sids, cache = [], [], 0
        for u in session[:-1]:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        cids.pop()
        sids.pop()
        # ground-truth
        ground_truth = session[-1]
        # negative samples
        ratio = random.random()
        if ratio > 0.5:
            # within session
            index = list(range(begin, max_l))
            index.remove(end)
            response = self.data[random.choice(index)]
        else:
            # random negative sample
            while True:
                rand_idx = random.randint(0, len(self.data)-1)
                if rand_idx != end:
                    break
            response = self.data[rand_idx]
        ratio = random.random()
        if ratio > 0.5:
            ids, tids, sids, cpids = self._packup(cids, ground_truth, response, sids)
            label = 1
        else:
            ids, tids, sids, cpids = self._packup(cids, response, ground_truth, sids)
            label = 0
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, tids, sids, cpids, mask_labels, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, sids, mask_labels, labels = [], [], [], [], []
        cpids = []
        for ids_, tids_, sids_, cpids_, mask_labels_, labels_ in batch:
            ids.append(ids_)
            sids.append(sids_)
            cpids.append(cpids_)
            tids.append(tids_)
            mask_labels.append(mask_labels_)
            labels.append(labels_)
        ids = [torch.LongTensor(i) for i in ids]
        sids = [torch.LongTensor(i) for i in sids]
        cpids = [torch.LongTensor(i) for i in cpids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        labels = torch.LongTensor(labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        cpids = pad_sequence(cpids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, sids, tids, cpids, mask_labels, attn_mask, labels = to_cuda(ids, sids, tids, cpids, mask_labels, attn_mask, labels)
        return {
            'ids': ids, 
            'sids': sids, 
            'tids': tids, 
            'cpids': cpids,
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }


class PostTrainMonoPlusDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_mono_plus_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # multiple sentences to append the post train corpus
        dataset = read_text_data_utterances(path, lang=args['lang'])
        dataset = [utterances for l, utterances in dataset if l == 1]
        data = []
        for utterances in dataset:
            for turn_l in range(self.args['min_turn_length'], self.args['max_turn_length']+1):
                turn_l_cands = []
                for s in range(max(0, len(utterances) - turn_l)):
                    turn_l_cands.append((s, s+turn_l))
                if len(turn_l_cands) > self.args['each_turn_max_sample_num']:
                    turn_l_cands = random.sample(turn_l_cands, self.args['each_turn_max_sample_num'])
                for s, e in turn_l_cands:
                    data.append(utterances[s:e])
        print(f'[!] collect {len(data)} samples for mono post training')

        self.data = []
        for utterances in tqdm(data):
            utterances = ' [SEP] '.join(utterances)
            item = self.vocab.encode(utterances, add_special_tokens=False)
            item = item[:self.args['max_len']-2]
            num_valid = len([i for i in item if i not in self.special_tokens])
            if num_valid < self.args['min_len']:
                continue
            self.data.append(item)
        print(f'[!] dataset size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tokens = self.data[i]
        ids = [self.cls] + tokens + [self.sep]
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, mask_labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.data)}')
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for ids_, mask_labels_ in batch:
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }

class PostTrainMonoSEEDDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_mono_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # for restoration-200k
        # for douban, ecommerce, ubuntu, restoration-200k just on their own dataset
        data = read_text_data_utterances(path, lang=args['lang'])
        data = list(chain(*[utterances for l, utterances in data if l == 1]))
        # also add the extended nonparallel corpus
        # in re-rank exp, do not use it; in full-rank exp, use it.
        if self.args['dataset'] in ['restoration-200k']:
            ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            data += read_extended_douban_corpus(ext_path)
        data = list(set(data))

        self.data = []
        for utterance in tqdm(data):
            item = self.vocab.encode(utterance, add_special_tokens=False)
            item = item[:self.args['max_len']-2]
            num_valid = len([i for i in item if i not in self.special_tokens])
            if num_valid < self.args['min_len']:
                continue
            self.data.append(item)
        print(f'[!] dataset size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tokens = self.data[i]
        ids = [self.cls] + tokens + [self.sep]
        no_mask_ids = deepcopy(ids)
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return no_mask_ids, ids, mask_labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.data)}')
        
    def collate(self, batch):
        no_mask_ids, ids, mask_labels = [], [], []
        for no_mask_ids_, ids_, mask_labels_ in batch:
            no_mask_ids.append(no_mask_ids_)
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        no_mask_ids = [torch.LongTensor(i) for i in no_mask_ids]
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        no_mask_ids = pad_sequence(no_mask_ids, batch_first=True, padding_value=self.pad)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        no_mask_ids, ids, mask_labels, attn_mask = to_cuda(no_mask_ids, ids, mask_labels, attn_mask)
        return {
            'no_mask_ids': no_mask_ids,
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }

        
class PostTrainMonoWriterDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask])

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] save the random access reader file into{rar_path}')
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        while True:
            sentences = json.loads(self.reader.get_line(i))['q']
            tokens = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
            ids = [self.cls]
            for item in tokens:
                ids.extend(item + [self.sep])
            ids = ids[:self.args['max_len']]
            if len([token for token in ids if token not in self.special_tokens]) > 5 + self.args['min_mask_num']:
                break
            else:
                # re-sampler a new item
                i = random.choice(range(0, self.size - 1))

        try:
            mask_labels = mask_sentence(
                ids,
                self.args['min_mask_num'], 
                self.args['max_mask_num'], 
                self.args['masked_lm_prob'], 
                special_tokens=self.special_tokens, 
                mask=self.mask, 
                vocab_size=len(self.vocab),
            )
        except:
            ipdb.set_trace()
        return ids, mask_labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.data)}')
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for ids_, mask_labels_ in batch:
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }

        
class WZPostTrainMonoDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask])
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccesReader Object over')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        while True:
            line = self.reader.get_line(i).strip()
            tokens = self.vocab.encode(line, add_special_tokens=False)[:self.args['max_len']]
            valid_token = [i for i in tokens if i not in self.special_tokens]
            # minimum length
            if len(valid_token) > self.args['min_len']:
                break
            else:
                i = random.choice(range(self.size))
        ids = [self.cls] + tokens + [self.sep]
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, mask_labels

    def save(self):
        pass
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for ids_, mask_labels_ in batch:
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }


# for bigger dataset that can not be loaded into the RAM
class PostTrainComparisonBigDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_rar.txt'
        path = f'{args["root_dir"]}/data/{args["dataset"]}/data.txt'
        self.reader = RandomAccessReader(path)
        self.reader.load_from_text(rar_path, size=10000)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] load RandomAccessReader over, dataset size: {self.size}')

    def __len__(self):
        return self.size
    
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
        try:
            line = json.loads(self.reader.get_line(i))
            session = self.vocab.batch_encode_plus(line['data'], add_special_tokens=False)['input_ids']
            cids = []
            for u in session[:-1]:
                cids.extend(u + [self.eos])
            cids.pop()
            # ground-truth
            ground_truth = session[-1]
            # negative samples
            ratio = random.random()
            if ratio > 0.7:
                response = random.choice(session[:-1])
            else:
                random_sample = json.loads(self.reader.get_line(random.choice(range(self.size))))['data']
                response = random.choice(random_sample)
                response = self.vocab.encode(response, add_special_tokens=False)
            ratio = random.random()
            if ratio > 0.5:
                ids, tids, cpids = self._packup(cids, ground_truth, response)
                label = 1
            else:
                ids, tids, cpids = self._packup(cids, response, ground_truth)
                label = 0
            mask_labels = mask_sentence(
                ids,
                self.args['min_mask_num'], 
                self.args['max_mask_num'], 
                self.args['masked_lm_prob'], 
                special_tokens=self.special_tokens, 
                mask=self.mask, 
                vocab_size=len(self.vocab),
            )
            return {
                'ids': ids, 
                'tids': tids, 
                'cpids': cpids, 
                'mask_labels': mask_labels, 
                'label': label
            }
        except Exception as error:
            print(error)
            return None

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, mask_labels, labels = [], [], [], []
        cpids = []
        for batch in batch:
            if batch is None:
                continue
            ids.append(batch['ids'])
            cpids.append(batch['cpids'])
            tids.append(batch['tids'])
            mask_labels.append(batch['mask_labels'])
            labels.append(batch['label'])
        ids = [torch.LongTensor(i) for i in ids]
        cpids = [torch.LongTensor(i) for i in cpids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        labels = torch.LongTensor(labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        cpids = pad_sequence(cpids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, cpids, mask_labels, attn_mask, labels = to_cuda(ids, tids, cpids, mask_labels, attn_mask, labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'cpids': cpids,
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }


class PostTrainMonoBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_rar.txt'
        path = f'{args["root_dir"]}/data/{args["dataset"]}/data.txt'
        self.reader = RandomAccessReader(path)
        self.reader.load_from_text(rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] load RandomAccessReader over, dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        try:
            line = json.loads(self.reader.get_line(i))
            sentences = line['data']
            max_l = -1
            for s in sentences:
                if len(s) > max_l:
                    max_l = len(s)
                    sentence = s

            ids = [self.cls] + self.vocab.encode(sentence, add_special_tokens=False)[:self.args['max_len']-2] + [self.sep]
            mask_label = mask_sentence(
                ids,
                self.args['min_mask_num'], 
                self.args['max_mask_num'], 
                self.args['masked_lm_prob'], 
                special_tokens=self.special_tokens, 
                mask=self.mask, 
                vocab_size=len(self.vocab),
            )
            return {
                'ids': ids,
                'mask_labels': mask_label
            }
        except:
            return None

    def save(self):
        pass
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for batch in batch:
            if batch is None:
                continue
            ids.append(batch['ids'])
            mask_labels.append(batch['mask_labels'])
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }


class PostTrainMonoPersonaChatDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_mono_persona_chat_{self.args["ext_read"]}_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_utterances(path, lang=args['lang'])
        self.data = []
        utterance_pool = set()
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            new_utterances = []
            for utterance in utterances:
                if '[split]' in utterance:
                    sub_utterances = utterance.split('[split]')
                    new_utterances.extend([u.strip() for u in sub_utterances])
                else:
                    new_utterances.append(utterance)
            new_utterances = list(set(new_utterances))
            new_utterances_v2 = []
            for u in new_utterances:
                if u in utterance_pool:
                    continue
                else:
                    utterance_pool.add(u)
                    new_utterances_v2.append(u)
            if len(new_utterances_v2) == 0:
                continue
            for utterance in new_utterances_v2:
                item = self.vocab.encode(utterance, add_special_tokens=False)
                item = item[:self.args['max_len']-2]
                num_valid = len([i for i in item if i not in self.special_tokens])
                if num_valid < self.args['min_len']:
                    continue
                self.data.append(item)
        print(f'[!] dataset size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tokens = self.data[i]
        ids = [self.cls] + tokens + [self.sep]
        mask_labels = mask_sentence(
            ids,
            self.args['min_mask_num'], 
            self.args['max_mask_num'], 
            self.args['masked_lm_prob'], 
            special_tokens=self.special_tokens, 
            mask=self.mask, 
            vocab_size=len(self.vocab),
        )
        return ids, mask_labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.data)}')
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for ids_, mask_labels_ in batch:
            ids.append(ids_)
            mask_labels.append(mask_labels_)
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }




