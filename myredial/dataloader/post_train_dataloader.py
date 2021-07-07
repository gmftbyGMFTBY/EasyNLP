from header import *
from .utils import *


class PostTrainDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        special_tokens_dict = {'eos_token': '[EOS]'}
        self.vocab.add_special_tokens(special_tokens_dict)

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
            for i in range(1, len(item)):
                if i < self.args['min_context_length']:
                    continue
                # begin, end, max-length session
                # check if the response is legal
                l = len([i for i in self.data[offset+i] if i not in self.special_tokens])
                if l > 0:
                    self.table.append((offset, offset+i, len(self.data)))

    def __len__(self):
        return len(self.table)

    def _mask_sentence(self, ids):
        mask_label = []
        mask_num = 0
        for i, t in enumerate(ids):
            if t in self.special_tokens:
                mask_label.append(-1)
            ratio = random.random()
            if ratio < 0.15:
                ratio /= 0.15
                if ratio < 0.8:
                    ids[i] = self.mask
                elif ratio < 0.9:
                    # random change
                    ids[i] = random.choice(list(range(self.vocab.vocab_size)))
                mask_label.append(t)
                mask_num += 1
            else:
                # not mask
                mask_label.append(-1)
        if mask_num < self.args['min_mask_num']:
            # at least mask one token
            mask_idx = random.choice(range(len(ids)))
            mask_label = [-1] * len(ids)
            mask_label[mask_idx] = ids[mask_idx]
            ids[mask_idx] = self.mask
        return mask_label

    def _truncate_pair(self, cids, rids, max_length):
        max_length -= 3    #  [CLS], [SEP], [SEP]
        while True:
            l = len(cids) + len(rids)
            if l <= max_length:
                break
            if len(cids) > 2 * len(rids):
                cids.pop(0)
            else:
                rids.pop()

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

        self._truncate_pair(tokens, response, self.args['max_len'])
        cids_mlm_label = self._mask_sentence(tokens)
        rids_mlm_label = self._mask_sentence(response)
        mask_labels = [-1] + cids_mlm_label + [-1] + rids_mlm_label + [-1]
        ids = [self.cls] + tokens + [self.sep] + response + [self.sep]
        tids = [0] * (len(tokens) + 2) + [1] * (len(response) + 1)
        return ids, tids, mask_labels, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
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
        mask_labels = pad_sequence(tids, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, tids, mask_labels, attn_mask, labels = ids.cuda(), tids.cuda(), mask_labels.cuda(), attn_mask.cuda(), labels.cuda()
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }
