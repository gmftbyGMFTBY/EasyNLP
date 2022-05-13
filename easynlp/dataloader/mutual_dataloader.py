from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualMutualFullDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.m_token = self.vocab.convert_tokens_to_ids('[M]')
        self.f_token = self.vocab.convert_tokens_to_ids('[F]')

        self.data = []
        path = path.replace('txt', 'pkl')
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_mutual(path)
            self.responses = []
            for context, response, hn_responses in tqdm(data):
                item = self.vocab.batch_encode_plus(context + [response] + hn_responses, add_special_tokens=False)['input_ids']
                context = item[:len(context)]
                response = item[len(context)]
                hn_responses = item[-len(hn_responses):]
                cids = []
                for u in context:
                    cids.extend(u + [self.sep])
                cids.pop()
                cids = [self.cls] + cids[-(self.args['max_len']-2):] + [self.sep]
                rids = [self.cls] + response[:(self.args['res_max_len']-2)] + [self.sep]
                hn_rids = [[self.cls] + hn[:self.args['res_max_len']-2] + [self.sep] for hn in hn_responses]
                self.data.append({
                    'ids': cids,
                    'rids': rids,
                    'hn_rids': hn_rids,
                })
                self.responses.append(rids)
            print(f'[!] load {len(self.responses)} utterances')
        else:
            data = read_text_data_utterances_test_mutual(path)
            for context, options, gt_index in tqdm(data):
                item = self.vocab.batch_encode_plus(context + options, add_special_tokens=False)['input_ids']
                context = item[:len(context)]
                options = item[-len(options):]
                ids = []
                for u in context:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = [self.cls] + ids[-(self.args['max_len']-2):] + [self.sep]
                rids = [[self.cls] + r[:(self.args['res_max_len']-2)] + [self.sep] for r in options]
                label = [1 if i == gt_index else 0 for i in range(4)]
                self.data.append({
                    'label': label,
                    'ids': ids,
                    'rids': rids
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            hn_rids = [torch.LongTensor(i) for i in bundle['hn_rids']]
            # if len(hn_rids) < self.args['gray_cand_num']:
            #     hn_rids.extend(random.sample(self.responses, self.args['gray_cand_num'] - len(hn_rids)))
            # else:
            #     hn_rids = random.sample(hn_rids, self.args['gray_cand_num'])
            return ids, rids, hn_rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            hn_rids = [i[2] for i in batch]
            return {
                'ids': ids, 
                'rids': rids, 
                'hn_rids': hn_rids,
            }
        else:
            assert len(batch) == 1
            ids, rids, label = batch[0]
            label = torch.LongTensor(label).cuda()
            return {
                'ids': ids, 
                'rids': rids, 
                'label': label,
            }


class PostTrainMonoMutualDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.m_token = self.vocab.convert_tokens_to_ids('[F]')
        self.f_token = self.vocab.convert_tokens_to_ids('[M]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos, self.m_token, self.f_token])

        path = path.replace('txt', 'pkl')
        data = read_text_data_utterances_mutual_mono(path)
        data = [u for u in list(set(data)) if u]
        print(f'[!] find {len(data)} sentences')

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


class BERTFTMutualDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.m_token = self.vocab.convert_tokens_to_ids('[M]')
        self.f_token = self.vocab.convert_tokens_to_ids('[F]')

        path = path.replace('txt', 'pkl')

        data = read_text_data_utterances_mutual_ft(path)
        self.data = []
        if self.args['mode'] == 'train':
            for context, options, gt_index in tqdm(data):
                item = self.vocab.batch_encode_plus(context + options, add_special_tokens=False)['input_ids']
                context = item[:len(context)]
                responses = item[-len(options):]
                ids = []
                for u in context:
                    ids.extend(u + [self.eos])
                ids.pop()

                a, b, c = [], [], []
                for idx, response in enumerate(responses):
                    context = deepcopy(ids)
                    truncate_pair(context, response, self.args['max_len'])
                    ids = [self.cls] + context + [self.sep] + response + [self.sep]
                    tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                    self.data.append({
                        'ids': ids, 
                        'tids': tids,
                        'label': 1 if idx == gt_index else 0,
                    })
        else:
            for context, options, gt_index in tqdm(data):
                item = self.vocab.batch_encode_plus(context + options, add_special_tokens=False)['input_ids']
                context = item[:len(context)]
                responses = item[-len(options):]
                ids = []
                for u in context:
                    ids.extend(u + [self.eos])
                ids.pop()

                a, b, c = [], [], []
                for idx, response in enumerate(responses):
                    context = deepcopy(ids)
                    truncate_pair(context, response, self.args['max_len'])
                    ids = [self.cls] + context + [self.sep] + response + [self.sep]
                    tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                    a.append(ids)
                    b.append(tids)
                    c.append(1 if idx == gt_index else 0)
                self.data.append({
                    'ids': a, 
                    'tids': b,
                    'label': c,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            label = bundle['label']
            return ids, tids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, tids, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
        else:
            ids, tids, label = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        label = torch.LongTensor(label)
        ids, tids, mask, label = to_cuda(ids, tids, mask, label)
        if self.args['mode'] == 'train':
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
            }


class PostTrainMutualDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.m_token = self.vocab.convert_tokens_to_ids('[M]')
        self.f_token = self.vocab.convert_tokens_to_ids('[F]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos, self.m_token, self.f_token])

        path = path.replace('txt', 'pkl')
        data = read_text_data_utterances_mutual_fp(path)
        self.data = []
        self.table = []
        for utterances in tqdm(data):
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            offset = len(self.data)
            self.data.extend(item)

            counter = 0
            l = []
            for utterance in item:
                l.append(len([i for i in utterance if i not in self.special_tokens]))
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
        pass
        
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


