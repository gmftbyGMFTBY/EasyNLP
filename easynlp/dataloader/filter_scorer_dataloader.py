from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class FilterPostTrainDataset(Dataset):

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
            utterances = [u for u in utterances if type(u) == str]
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
            return None, None, None, None
        return ids, tids, mask_labels, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, mask_labels, labels = [], [], [], []
        for ids_, tids_, mask_labels_, labels_ in batch:
            if ids_ is None:
                continue
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


class FilterDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.vocab.add_tokens(['[EOS]'])
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.sep = self.vocab.sep_token_id

        if self.args['mode'] == 'train':
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccessReader Object over')
            else:
                self.reader = RandomAccessReader(path)
                self.reader.init()
                torch.save(self.reader, rar_path)
            self.size = self.reader.size
            self.reader.init_file_handler()
            print(f'[!] dataset size: {self.size}')
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            self.data = []
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids, context, responses = [], [], [], []
                for b in batch:
                    label = b[0]
                    utterances = b[1]
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
                    'responses': responses,
                })
            self.size = len(self.data)
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i).strip()
            # item = json.loads(line)
            items = line.strip().split('\t')
            item = {
                'score': int(items[0]),
                'data': items[1:]
            }
            label = int(item['score'])
            if label not in [0, 1]:
                return None, None, None
            data = item['data']
            try:
                items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
                response = items[-1]
                context = []
                for u in items[:-1]:
                    context.extend(u + [self.sep])
                context.pop()
                truncate_pair(context, response, self.args['max_len'])
            except:
                return None, None, None
            ids = [self.cls] + context + [self.sep] + response + [self.sep]
            tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
            return ids, tids, label
        else:
            bundle = self.data[i]
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [], [], []
            for a, b, c in batch:
                if a and b:
                    ids.append(torch.LongTensor(a))
                    tids.append(torch.LongTensor(b))
                    label.append(c)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, tids, ids_mask, label = to_cuda(ids, tids, ids_mask, label)
            return {
                'ids': ids, 
                'tids': tids,
                'mask': ids_mask, 
                'label': label
            }
        else:
            assert len(batch) == 1
            ids, tids, label, context, responses = batch[0]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, tids, ids_mask, label = to_cuda(ids, tids, ids_mask, label)
            return {
                'ids': ids,
                'tids': tids,
                'mask': ids_mask,
                'label': label,
                'context': context,
                'responses': responses
            }


class FilterDRDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        item = json.loads(line)
        if self.args['mode'] == 'train':
            data = item['data']
            try:
                items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            except:
                return None, None
            response = items[-1]
            context = []
            for u in items[:-1]:
                context.extend(u + [self.sep])
            context.pop()
            context = context[-self.args['max_len']:]
            response = response[:self.args['res_max_len']]
            ids = [self.cls] + context + [self.sep]
            rids = [self.cls] + response + [self.sep]
            return ids, rids
        else:
            data = item['data'][:-1]
            candidates = [item['data'][-1]] + item['neg_sentences']
            items = self.vocab.batch_encode_plus(data, add_special_tokens=False)['input_ids']
            context = []
            for u in items:
                context.extend(u + [self.sep])
            context.pop() 
            c = context[-self.args['max_len']:]
            c = [self.cls] + c + [self.sep]
            items = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']

            assert len(items) == 10
            rids, label = [], []
            for idx, utter in enumerate(items):
                if idx == 0:
                    label.append(1)
                else:
                    label.append(0)
                r = deepcopy(utter)
                r = r[:self.args['res_max_len']]
                ids_ = [self.cls] + c + [self.sep]
                rids_ = [self.cls] + r + [self.sep]
                rids.append(rids_)
            return c, rids, label

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [], []
            for a, b in batch:
                if a and b:
                    ids.append(torch.LongTensor(a))
                    rids.append(torch.LongTensor(b))
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids, pad_token_idx=self.pad)
            rids_mask = generate_mask(rids, pad_token_idx=self.pad)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids,
                'ids_mask': ids_mask,
                'rids_mask': rids_mask,
            }
        else:
            assert len(batch) == 1
            ids, rids, label = batch[0]
            ids = torch.LongTensor(ids)
            rids = pad_sequence([torch.LongTensor(i) for i in rids], batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids, pad_token_idx=self.pad)
            label = torch.LongTensor(label)
            ids, rids, rids_mask = to_cuda(ids, rids, rids_mask)
            label = label.cuda()
            return {
                'ids': ids,
                'rids': rids,
                'rids_mask': rids_mask,
                'label': label,
            }


class FilterSensitiveTestDataset(Dataset):

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
        self.pp_path = f'{os.path.splitext(path)[0]}_sensitive_test_train_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data, self.table = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        self.table = []
        for label, utterances in tqdm(data):
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ctx, res = item[:-1], item[-1]
            ids = []
            for u in ctx:
                ids.extend(u + [self.eos])
            ids.pop()
            truncate_pair(ids, res, self.args['max_len'])
            ids = [self.cls] + ids + [self.sep] + res + [self.sep]
            tids = [0] * (len(ids) + 2) + [1] * (len(res) + 1)
            self.data.append({
                'label': label,
                'ids': ids,
                'tids': tids
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        b = self.data[i]
        return b['ids'], b['tids'], b['label']

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, labels = [], [], []
        for a, b, c in batch:
            ids.append(torch.LongTensor(a))
            tids.append(torch.LongTensor(b))
            labels.append(c)
        labels = torch.LongTensor(labels)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        attn_mask = generate_mask(ids)
        ids, tids, attn_mask, labels = to_cuda(ids, tids, attn_mask, labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'attn_mask': attn_mask, 
            'label': labels,
        }
