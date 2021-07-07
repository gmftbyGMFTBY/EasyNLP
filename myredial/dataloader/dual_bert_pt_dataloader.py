from header import *
from .utils import *

class BERTDualPTDataset(Dataset):

    '''Dual bert dataloader with post train (MLM)'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        if self.args['mode'] == 'train':
            self.masked_lm_prob = self.args['masked_lm_prob']
            self.min_masked_lm_prob = self.args['min_masked_lm_prob']
            self.max_prediction_token = self.args['max_prediction_token']
            self.min_prediction_token = self.args['min_prediction_token']
            self.context_valid_token_num = self.args['context_valid_token_num']
            self.response_valid_token_num = self.args['response_valid_token_num']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_post_train_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_dual_bert(path, lang=self.args['lang'])

        self.data = []
        if self.args['mode'] == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context, response])
                ids, rids = item['input_ids'][0], item['input_ids'][1]
                ids, rids = self._length_limit(ids), self._length_limit_res(rids)

                # check the data
                valid_tokens_num = len([i for i in rids if i not in [self.sep, self.cls, self.unk]])
                if valid_tokens_num < self.response_valid_token_num:
                    continue
                valid_tokens_num = len([i for i in ids if i not in [self.sep, self.cls, self.unk]])
                if valid_tokens_num < self.context_valid_token_num:
                    continue

                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': context,
                    'rtext': response,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for item_ in batch:
                    item = self.vocab.batch_encode_plus([item_[1], item_[2]])
                    ids = item['input_ids'][0]
                    rids.append(item['input_ids'][1])
                    if item_[0] == 1:
                        gt_text.append(item_[2])
                ids, rids = self._length_limit(ids), [self._length_limit_res(rids_) for rids_ in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    

    def _set_mlm_prob_delta(self, total_step):
        self.masked_lm_prob_delta = (self.masked_lm_prob - self.min_masked_lm_prob) / total_step

    def _update_mlm_prob_delta(self):
        # update the masked_lm_prob
        self.masked_lm_prob -= self.masked_lm_prob_delta

    def _mask_sentence(self, ids):
        indexes = [idx for idx, i in enumerate(ids) if i not in [self.cls, self.sep, self.unk, self.mask]]
        valid_tokens_num = len(indexes)
        num_pred = max(
            self.min_prediction_token,
            min(
                self.max_prediction_token,
                int(self.masked_lm_prob * valid_tokens_num),
            )
        )

        if num_pred > valid_tokens_num:
            mask_label = [-1] * len(ids)
            return ids, mask_label

        mask_idx = random.sample(indexes, num_pred)
        n_ids, mask_label = [], []

        for i in range(len(ids)):
            token = ids[i]
            if i in mask_idx:
                ratio = random.random()
                if ratio < 0.8:
                    # mask
                    n_ids.append(self.mask)
                    mask_label.append(token)
                elif ratio < 0.9:
                    n_ids.append(random.choice(list(range(self.vocab.vocab_size))))
                    mask_label.append(token)
                else:
                    n_ids.append(token)
                    mask_label.append(token)
            else:
                n_ids.append(token)
                mask_label.append(-1)
        return n_ids, mask_label
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            (ids1, ids1_mask_label), (ids2, ids2_mask_label) = self._mask_sentence(bundle['ids']), self._mask_sentence(bundle['ids'])
            (rids1, rids1_mask_label), (rids2, rids2_mask_label) = self._mask_sentence(bundle['rids']), self._mask_sentence(bundle['rids'])
            return ids1, ids2, rids1, rids2, ids1_mask_label, ids2_mask_label, rids1_mask_label, rids2_mask_label, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; dataset size: {len(self.data)}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids1, ids2, rids1, rids2, ids1_mask_label, ids2_mask_label, rids1_mask_label, rids2_mask_label, ctext, rtext = [], [], [], [], [], [], [], [], [], []
            for a, b, c, d, e, f, g, h, i, j in batch:
                ids1.append(a)
                ids2.append(b)
                rids1.append(c)
                rids2.append(d)
                ids1_mask_label.append(e)
                ids2_mask_label.append(f)
                rids1_mask_label.append(g)
                rids2_mask_label.append(h)
                ctext.append(i)
                rtext.append(j)

            ids = [torch.LongTensor(i) for i in ids1 + ids2]
            rids = [torch.LongTensor(i) for i in rids1 + rids2]
            ids_mask_label = [torch.LongTensor(i) for i in ids1_mask_label + ids2_mask_label]
            rids_mask_label = [torch.LongTensor(i) for i in rids1_mask_label + rids2_mask_label]

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask_label = pad_sequence(ids_mask_label, batch_first=True, padding_value=-1)
            rids_mask_label = pad_sequence(rids_mask_label, batch_first=True, padding_value=-1)
            ids_mask = self.generate_mask(ids)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                ids, rids, ids_mask, rids_mask = ids.cuda(), rids.cuda(), ids_mask.cuda(), rids_mask.cuda()

            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ids_mask_label': ids_mask_label,
                'rids_mask_label': rids_mask_label,
                'ctext': ctext,
                'rtext': rtext,
                'masked_lm_prob': self.masked_lm_prob,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                ids, rids, rids_mask, label = ids.cuda(), rids.cuda(), rids_mask.cuda(), label.cuda()
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }
