from header import *
from .utils import *
from .util_func import *

class BERTDualPTDataset(Dataset):

    '''Dual bert dataloader for post train (MLM)'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_post_train_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = item[:-1], item[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]
            ids = [self.cls] + ids + [self.sep]
            rids = rids[:(self.args['res_max_len']-2)]
            rids = [self.cls] + rids + [self.sep]
            num_valid_ctx = len([i for i in ids if i not in self.special_tokens])
            num_valid_res = len([i for i in rids if i not in self.special_tokens])
            if num_valid_ctx < self.args['min_ctx_len'] or num_valid_res < self.args['min_res_len']:
                continue
            self.data.append({
                'ids': ids,
                'rids': rids,
                'ctext': utterances[:-1],
                'rtext': utterances[-1],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids, rids = bundle['ids'], bundle['rids']
        mask_labels_ids = mask_sentence(
            ids,
            self.args['min_mask_num'],
            self.args['max_mask_num'],
            self.args['masked_lm_prob'],
            special_tokens=self.special_tokens,
            mask=self.mask,
            vocab_size=len(self.vocab)
        )
        mask_labels_rids = mask_sentence(
            rids,
            self.args['min_mask_num'],
            self.args['max_mask_num'],
            self.args['masked_lm_prob'],
            special_tokens=self.special_tokens,
            mask=self.mask,
            vocab_size=len(self.vocab)
        )
        return ids, mask_labels_ids, rids, mask_labels_rids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; dataset size: {len(self.data)}')
        
    def collate(self, batch):
        ids, mask_labels_ids, rids, mask_labels_rids = [], [], [], []
        for a, b, c, d in batch:
            ids.append(torch.LongTensor(a))
            mask_labels_ids.append(torch.LongTensor(b))
            rids.append(torch.LongTensor(c))
            mask_labels_rids.append(torch.LongTensor(d))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        rids_mask = generate_mask(rids)
        mask_labels_ids = pad_sequence(mask_labels_ids, batch_first=True, padding_value=self.pad)
        mask_labels_rids = pad_sequence(mask_labels_rids, batch_first=True, padding_value=self.pad)
        ids, ids_mask, mask_labels_ids, rids, rids_mask, mask_labels_rids = to_cuda(ids, ids_mask, mask_labels_ids, rids, rids_mask, mask_labels_ids)
        return {
            'ids': ids,
            'ids_mask': ids_mask,
            'mask_labels_ids': mask_labels_ids,
            'rids': rids,
            'rids_mask': rids_mask,
            'mask_labels_rids': mask_labels_rids,
        }
