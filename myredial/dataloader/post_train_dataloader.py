from header import *
from .utils import *


class PostTrainDataset(Dataset):

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

        suffix = args['tokenizer'].replace('/', '_') + '_wwm'
        if self.args['wwm']:
            suffix += '_wwm'
        self.pp_path = f'{os.path.splitext(path)[0]}_post_train_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        if self.args['wwm']:
            data = read_text_data_utterances_wwm(path, lang=self.args['lang'])
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        for label, utterances in tqdm(data):
            label = 2 if label == 1 else 0
            context = ' [SEP] '.join(utterances[:-1])
            response = utterances[-1]
            item = self.text_to_ids([context, response]+utterances[:-1])
            cws, rws, hws = item[0], item[1], item[2:]
            hws = [self._length_limit_res(i) for i in hws]

            cws, rws = self._length_limit(cws, rws)
            self.data.append({'label': label, 'cws': cws, 'rws': rws, 'hws': hws})

    def text_to_ids(self, texts, lang='zh'):
        rest = []
        for text in texts:
            if lang == 'en':
                rest.append(
                    ['[CLS]'] + self.vocab.tokenize(text) + ['[SEP]']
                )
            elif lang == 'zh': 
                if self.args['wwm']:
                    # add ## speical tokens before the sub-word in chinese
                    text = text.replace(' ', ' [SEP] ')
                    tokens = self.vocab.tokenize(text)
                    new_tokens = []
                    is_sub = False
                    for token in tokens:
                        if token == '[SEP]':
                            is_sub = False
                            continue
                        if is_sub:
                            if not token.startswith('##'):
                                token = f'##{token}'
                            new_tokens.append(token)
                        else:
                            new_tokens.append(token)
                            is_sub = True
                    tokens = ['[CLS]'] + new_tokens + ['[SEP]']
                    rest.append(tokens)
                else:
                    rest.append(
                        ['[CLS]'] + self.vocab.tokenize(text) + ['[SEP]']
                    )
        return rest

    def _length_limit(self, cids, rids):
        # cids
        if len(cids) > self.args['max_len']:
            cids = [cids[0]] + cids[-(self.args['max_len']-1):]     # [CLS] ... [SEP]
        # rids: without [CLS] token
        rids = self._length_limit_res(rids)
        return cids, rids

    def _length_limit_res(self, rids):
        if len(rids) > self.args['res_max_len']:
            rids = rids[1:self.args['res_max_len']] + [self.sep] 
        else:
            rids = rids[1:]
        return rids
                
    def __len__(self):
        return len(self.data)

    def _get_ids(self, token):
        if token.startswith('##'):
            token = token.lstrip('##')
        return self.vocab.convert_tokens_to_ids(token)

    def _mask_sentence(self, words):
        cand = []    # ([mask_label_ids1, mask_label_ids2, ...], [ids1, ids2, ...])
        ipdb.set_trace()
        for i, token in enumerate(words):
            # special tokens
            if token in ['[CLS]', '[SEP]', '[UNK]']:
                cand.append(([-1], [self._get_ids(token)]))
                continue

            ratio = random.random()
            if ratio < 0.15:
                ratio /= 0.15
                if ratio < 0.8:
                    ipt = [-1]
                elif ratio < 0.9:
                    # random change
                    ipt = [random.choice(list(range(self.vocab.vocab_size)))]
                else:
                    ipt = self._get_ids(token)
                l = [self._get_ids(token)]
                if self.args['wwm'] and token.startswith('##'):
                    cand[-1] = (cand[-1][0] + l, cand[-1][1] + ipt)
                else:
                    cand.append((l, ipt))
            else:
                # not mask
                cand.append(([-1], [self._get_ids(token)]))
        
        mask_label = []
        token_ids = []
        for opt, ipt in cand:
            for i, j in zip(opt, ipt):
                mask_label.append(i)
                token_ids.append(j)
        return token_ids, mask_label

    def _packup(self, cws, rws):
        '''generate the token_type_ids, ids, and the mask label'''
        cids, cids_mask = self._mask_sentence(cws)
        rids, rids_mask = self._mask_sentence(rws)
        ids = cids + rids
        tids = [1] * len(cids) + [0] * len(rids)
        mask_label = cids_mask + rids_mask
        return ids, tids, mask_label 

    def __getitem__(self, i):
        bundle = self.data[i]

        ids, tids, mask_labels, labels = [], [], [], []
        # label 2 or 0
        cws, rws = bundle['cws'], bundle['rws']
        ids_, tids_, mask_label_ = self._packup(cws, rws)
        label = bundle['label']
        ids.append(ids_)
        tids.append(tids_)
        mask_labels.append(mask_label_)
        labels.append(label)

        # label: 1
        hws = random.choice(bundle['hws'])
        ids_, tids_, mask_label_ = self._packup(cids, hws)
        ids.append(ids_)
        tids.append(tids_)
        mask_labels.append(mask_label_)
        labels.append(1)
        return ids, tids, mask_labels, labels

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
        ids, tids, mask_labels, labels = [], [], [], []
        for ids_, tids_, mask_labels_, labels_ in batch:
            ids.extend(ids_)
            tids.extend(tids_)
            mask_labels.extend(mask_labels_)
            labels.extend(labels_)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(tids, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, mask_labels, attn_mask, label = ids.cuda(), tids.cuda(), mask_labels.cuda(), attn_mask.cuda(), label.cuda()
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': label,
        }
