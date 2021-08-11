from header import *
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
                if sum(l[:i+1]) > self.args['min_token_length'] and l[i] > 0:
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
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_mono_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_extended_douban_corpus(path)
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
