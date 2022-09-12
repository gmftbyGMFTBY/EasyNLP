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


class BERTCompareMutualDataset(Dataset):
    
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
                    if idx != gt_index:
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[gt_index])
                        nt_response = deepcopy(responses[idx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        
                        ids_1 = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        label_1 = 0

                        ids_2 = [self.cls] + context + [self.sep] + nt_response + [self.sep] + gt_response + [self.sep]
                        tids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + len(gt_response) + 2)
                        pids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + 1) + [2] * (len(gt_response) + 1)
                        label_2 = 1
                        self.data.append({
                            'ids': ids_1, 
                            'tids': tids_1,
                            'pids': pids_1,
                            'label': label_1,
                        })
                        self.data.append({
                            'ids': ids_2, 
                            'tids': tids_2,
                            'pids': pids_2,
                            'label': label_2,
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

                a, b, d = [], [], []
                for idx, response_i in enumerate(responses):
                    for jdx, response_j in enumerate(responses):
                        if idx == jdx:
                            continue
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[idx])
                        nt_response = deepcopy(responses[jdx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        ids_ = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        a.append(ids_)
                        b.append(tids_)
                        d.append(pids_)
                c = [1 if i == gt_index else 0 for i in range(4)]
                self.data.append({
                    'ids': a, 
                    'tids': b,
                    'pids': d,
                    'label': c,
                })

    def truncate_triple(self, cids, rids1, rids2, max_length):
        max_length -= 4
        while True:
            l = len(cids) + len(rids1) + len(rids2)
            if l <= max_length:
                break
            if len(cids) > len(rids1) + len(rids2):
                cids.pop(0)
            elif len(rids1) > len(rids2):
                rids1.pop()
            else:
                rids2.pop()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            pids = torch.LongTensor(bundle['pids'])
            label = bundle['label']
            return ids, tids, pids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            pids = [torch.LongTensor(i) for i in bundle['pids']]
            return ids, tids, pids, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, pids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
        else:
            ids, tids, pids, label = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        pids = pad_sequence(pids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids, pad_token_idx=self.pad)
        label = torch.LongTensor(label)
        ids, tids, pids, mask, label = to_cuda(ids, tids, pids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'pids': pids,
            'mask': mask, 
            'label': label
        }


class BERTCompareMutualV3Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

        # self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        # self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        # self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        # self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.pad = self.vocab.convert_tokens_to_ids('<pad>')
        self.sep = self.vocab.convert_tokens_to_ids('</s>')
        self.cls = self.vocab.convert_tokens_to_ids('<s>')
        self.unk = self.vocab.convert_tokens_to_ids('<unk>')

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
                    # ids.extend(u + [self.eos])
                    ids.extend(u)
                ids.pop()

                a, b, c = [], [], []
                for idx, response in enumerate(responses):
                    if idx != gt_index:
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[gt_index])
                        nt_response = deepcopy(responses[idx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        
                        ids_1 = [self.cls] + context + [self.eos] + gt_response + [self.eos] + nt_response + [self.sep]
                        tids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        label_1 = 0
                        first_index_1 = len(context) + 1
                        second_index_1 = len(context) + 2 + len(gt_response)

                        ids_2 = [self.cls] + context + [self.eos] + nt_response + [self.eos] + gt_response + [self.sep]
                        tids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + len(gt_response) + 2)
                        pids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + 1) + [2] * (len(gt_response) + 1)
                        label_2 = 1
                        first_index_2 = len(context) + 1
                        second_index_2 = len(context) + 2 + len(nt_response)

                        # label 2
                        context = deepcopy(ids)
                        nt_response = deepcopy(responses[idx])
                        idx_ = random.choice([i for i in range(len(responses)) if i != gt_index and i != idx])
                        # idx_ = random.choice([i for i in range(len(responses)) if i != gt_index])
                        nt2_response = deepcopy(responses[idx_])
                        self.truncate_triple(context, nt2_response, nt_response, self.args['max_len'])

                        ids_3 = [self.cls] + context + [self.eos] + nt2_response + [self.eos] + nt_response + [self.sep]
                        tids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + len(nt_response) + 2)
                        pids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + 1) + [2] * (len(nt_response) + 1)
                        label_3 = 2
                        first_index_3 = len(context) + 1
                        second_index_3 = len(context) + 2 + len(nt2_response)

                        self.data.append({
                            'ids': ids_1, 
                            'tids': tids_1,
                            'pids': pids_1,
                            'label': label_1,
                            'first_index': first_index_1,
                            'second_index': second_index_1,
                        })
                        self.data.append({
                            'ids': ids_2, 
                            'tids': tids_2,
                            'pids': pids_2,
                            'label': label_2,
                            'first_index': first_index_2,
                            'second_index': second_index_2,
                        })
                        # self.data.append({
                        #     'ids': ids_3, 
                        #     'tids': tids_3,
                        #     'pids': pids_3,
                        #     'label': label_3,
                        #     'first_index': first_index_3,
                        #     'second_index': second_index_3,
                        # })
        else:
            for context, options, gt_index in tqdm(data):
                item = self.vocab.batch_encode_plus(context + options, add_special_tokens=False)['input_ids']
                context = item[:len(context)]
                responses = item[-len(options):]
                ids = []
                for u in context:
                    ids.extend(u)
                ids.pop()

                a, b, d = [], [], []
                ff, ss = [], []
                for idx, response_i in enumerate(responses):
                    for jdx, response_j in enumerate(responses):
                        if idx == jdx:
                            continue
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[idx])
                        nt_response = deepcopy(responses[jdx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        ids_ = [self.cls] + context + [self.eos] + gt_response + [self.eos] + nt_response + [self.sep]
                        tids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        first_index = len(context) + 2
                        second_index = len(context) + 2 + len(gt_response)
                        a.append(ids_)
                        b.append(tids_)
                        d.append(pids_)
                        ff.append(first_index)
                        ss.append(second_index)
                c = [1 if i == gt_index else 0 for i in range(4)]
                self.data.append({
                    'ids': a, 
                    'tids': b,
                    'pids': d,
                    'label': c,
                    'first_index': ff,
                    'second_index': ss
                })

    def truncate_triple(self, cids, rids1, rids2, max_length):
        max_length -= 4
        while True:
            l = len(cids) + len(rids1) + len(rids2)
            if l <= max_length:
                break
            if len(cids) > len(rids1) + len(rids2):
                cids.pop(0)
            elif len(rids1) > len(rids2):
                rids1.pop()
            else:
                rids2.pop()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            pids = torch.LongTensor(bundle['pids'])
            label = bundle['label']
            return ids, tids, pids, label, bundle['first_index'], bundle['second_index']
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            pids = [torch.LongTensor(i) for i in bundle['pids']]
            return ids, tids, pids, bundle['label'], bundle['first_index'], bundle['second_index']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, pids, label, first_index, second_index = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch], [i[4] for i in batch], [i[5] for i in batch]
        else:
            ids, tids, pids, label, first_index, second_index = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=0)
        pids = pad_sequence(pids, batch_first=True, padding_value=self.pad)
        first_index = torch.LongTensor(first_index)
        second_index = torch.LongTensor(second_index)
        mask = generate_mask(ids, pad_token_idx=self.pad)
        label = torch.LongTensor(label)
        ids, tids, pids, mask, label, first_index, second_index = to_cuda(ids, tids, pids, mask, label, first_index, second_index)
        return {
            'ids': ids, 
            'tids': tids, 
            'pids': pids,
            'mask': mask, 
            'label': label,
            'first_index': first_index,
            'second_index': second_index,
        }



class BERTCompareMutualV4Dataset(Dataset):

    '''add the easy training samples'''
    
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
                dialog_history = context
                ids = []
                for u in context:
                    ids.extend(u + [self.eos])
                ids.pop()

                a, b, c = [], [], []
                for idx, response in enumerate(responses):
                    if idx != gt_index:
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[gt_index])
                        nt_response = deepcopy(responses[idx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        
                        ids_1 = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        label_1 = 0

                        ids_2 = [self.cls] + context + [self.sep] + nt_response + [self.sep] + gt_response + [self.sep]
                        tids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + len(gt_response) + 2)
                        pids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + 1) + [2] * (len(gt_response) + 1)
                        label_2 = 1

                        # label 2
                        context = deepcopy(ids)
                        nt_response = deepcopy(responses[idx])
                        idx_ = random.choice([i for i in range(len(responses)) if i != gt_index and i != idx])
                        nt2_response = deepcopy(responses[idx_])
                        self.truncate_triple(context, nt2_response, nt_response, self.args['max_len'])

                        ids_3 = [self.cls] + context + [self.sep] + nt2_response + [self.sep] + nt_response + [self.sep]
                        tids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + len(nt_response) + 2)
                        pids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + 1) + [2] * (len(nt_response) + 1)
                        label_3 = 2

                        self.data.append({
                            'ids': ids_1, 
                            'tids': tids_1,
                            'pids': pids_1,
                            'label': label_1,
                        })
                        self.data.append({
                            'ids': ids_2, 
                            'tids': tids_2,
                            'pids': pids_2,
                            'label': label_2,
                        })
                        self.data.append({
                            'ids': ids_3, 
                            'tids': tids_3,
                            'pids': pids_3,
                            'label': label_3,
                        })
                    else:
                        # ground_truth vs. easy negative samples
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[gt_index])
                        nt_response = deepcopy(random.choice(dialog_history))
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])

                        ids_1 = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        label_1 = 0
                        
                        ids_2 = [self.cls] + context + [self.sep] + nt_response + [self.sep] + gt_response + [self.sep]
                        tids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + len(gt_response) + 2)
                        pids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + 1) + [2] * (len(gt_response) + 1)
                        label_2 = 1

                        context = deepcopy(ids)
                        nt_response = deepcopy(random.choice(dialog_history))
                        nt2_response = deepcopy(random.choice(dialog_history))
                        self.truncate_triple(context, nt2_response, nt_response, self.args['max_len'])

                        ids_3 = [self.cls] + context + [self.sep] + nt2_response + [self.sep] + nt_response + [self.sep]
                        tids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + len(nt_response) + 2)
                        pids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + 1) + [2] * (len(nt_response) + 1)
                        label_3 = 2

                        self.data.append({
                            'ids': ids_1, 
                            'tids': tids_1,
                            'pids': pids_1,
                            'label': label_1,
                        })
                        self.data.append({
                            'ids': ids_2, 
                            'tids': tids_2,
                            'pids': pids_2,
                            'label': label_2,
                        })
                        self.data.append({
                            'ids': ids_3, 
                            'tids': tids_3,
                            'pids': pids_3,
                            'label': label_3,
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

                a, b, d = [], [], []
                for idx, response_i in enumerate(responses):
                    for jdx, response_j in enumerate(responses):
                        if idx == jdx:
                            continue
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[idx])
                        nt_response = deepcopy(responses[jdx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        ids_ = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_ = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        a.append(ids_)
                        b.append(tids_)
                        d.append(pids_)
                c = [1 if i == gt_index else 0 for i in range(4)]
                self.data.append({
                    'ids': a, 
                    'tids': b,
                    'pids': d,
                    'label': c,
                })

    def truncate_triple(self, cids, rids1, rids2, max_length):
        max_length -= 4
        while True:
            l = len(cids) + len(rids1) + len(rids2)
            if l <= max_length:
                break
            if len(cids) > len(rids1) + len(rids2):
                cids.pop(0)
            elif len(rids1) > len(rids2):
                rids1.pop()
            else:
                rids2.pop()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            pids = torch.LongTensor(bundle['pids'])
            label = bundle['label']
            return ids, tids, pids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            pids = [torch.LongTensor(i) for i in bundle['pids']]
            return ids, tids, pids, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, pids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
        else:
            ids, tids, pids, label = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        pids = pad_sequence(pids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids, pad_token_idx=self.pad)
        label = torch.LongTensor(label)
        ids, tids, pids, mask, label = to_cuda(ids, tids, pids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'pids': pids,
            'mask': mask, 
            'label': label
        }



class BERTCompareMutualV3PostTrainDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        # self.pad = self.vocab.convert_tokens_to_ids('<pad>')
        # self.sep = self.vocab.convert_tokens_to_ids('</s>')
        # self.cls = self.vocab.convert_tokens_to_ids('<s>')
        # self.unk = self.vocab.convert_tokens_to_ids('<unk>')
        # self.mask = self.vocab.convert_tokens_to_ids('<mask>')

        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.m_token = self.vocab.convert_tokens_to_ids('[M]')
        self.f_token = self.vocab.convert_tokens_to_ids('[F]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos, self.m_token, self.f_token])

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
                    if idx != gt_index:
                        context = deepcopy(ids)
                        gt_response = deepcopy(responses[gt_index])
                        nt_response = deepcopy(responses[idx])
                        self.truncate_triple(context, gt_response, nt_response, self.args['max_len'])
                        
                        ids_1 = [self.cls] + context + [self.sep] + gt_response + [self.sep] + nt_response + [self.sep]
                        tids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + len(nt_response) + 2)
                        pids_1 = [0] * (len(context) + 2) + [1] * (len(gt_response) + 1) + [2] * (len(nt_response) + 1)
                        label_1 = 0
                        first_index_1 = len(context) + 1
                        second_index_1 = len(context) + 2 + len(gt_response)

                        ids_2 = [self.cls] + context + [self.sep] + nt_response + [self.sep] + gt_response + [self.sep]
                        tids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + len(gt_response) + 2)
                        pids_2 = [0] * (len(context) + 2) + [1] * (len(nt_response) + 1) + [2] * (len(gt_response) + 1)
                        label_2 = 1
                        first_index_2 = len(context) + 1
                        second_index_2 = len(context) + 2 + len(nt_response)

                        # label 2
                        context = deepcopy(ids)
                        nt_response = deepcopy(responses[idx])
                        idx_ = random.choice([i for i in range(len(responses)) if i != gt_index and i != idx])
                        nt2_response = deepcopy(responses[idx_])
                        self.truncate_triple(context, nt2_response, nt_response, self.args['max_len'])

                        ids_3 = [self.cls] + context + [self.sep] + nt2_response + [self.sep] + nt_response + [self.sep]
                        tids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + len(nt_response) + 2)
                        pids_3 = [0] * (len(context) + 2) + [1] * (len(nt2_response) + 1) + [2] * (len(nt_response) + 1)
                        label_3 = 2
                        first_index_3 = len(context) + 1
                        second_index_3 = len(context) + 2 + len(nt2_response)

                        self.data.append({
                            'ids': ids_1, 
                            'tids': tids_1,
                            'pids': pids_1,
                            'label': label_1,
                            'first_index': first_index_1,
                            'second_index': second_index_1,
                        })
                        self.data.append({
                            'ids': ids_2, 
                            'tids': tids_2,
                            'pids': pids_2,
                            'label': label_2,
                            'first_index': first_index_2,
                            'second_index': second_index_2,
                        })
                        self.data.append({
                            'ids': ids_3, 
                            'tids': tids_3,
                            'pids': pids_3,
                            'label': label_3,
                            'first_index': first_index_3,
                            'second_index': second_index_3,
                        })

    def truncate_triple(self, cids, rids1, rids2, max_length):
        max_length -= 4
        while True:
            l = len(cids) + len(rids1) + len(rids2)
            if l <= max_length:
                break
            if len(cids) > len(rids1) + len(rids2):
                cids.pop(0)
            elif len(rids1) > len(rids2):
                rids1.pop()
            else:
                rids2.pop()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        mask_labels = mask_sentence(
            bundle['ids'],
            self.args['min_mask_num'],
            self.args['max_mask_num'],
            self.args['masked_lm_prob'],
            special_tokens=self.special_tokens,
            mask=self.mask,
            vocab_size=len(self.vocab)
        )
        ids = torch.LongTensor(bundle['ids'])
        tids = torch.LongTensor(bundle['tids'])
        pids = torch.LongTensor(bundle['pids'])
        mask_labels = torch.LongTensor(mask_labels)
        label = bundle['label']
        return ids, tids, pids, label, mask_labels, bundle['first_index'], bundle['second_index']

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, pids, label, mask_labels, first_index, second_index = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch], [i[4] for i in batch], [i[5] for i in batch], [i[6] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=0)
        pids = pad_sequence(pids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)
        first_index = torch.LongTensor(first_index)
        second_index = torch.LongTensor(second_index)
        mask = generate_mask(ids, pad_token_idx=self.pad)
        label = torch.LongTensor(label)
        ids, tids, pids, mask, label, first_index, second_index, mask_labels = to_cuda(ids, tids, pids, mask, label, first_index, second_index, mask_labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'pids': pids,
            'mask': mask, 
            'label': label,
            'mask_labels': mask_labels,
            'first_index': first_index,
            'second_index': second_index,
        }


