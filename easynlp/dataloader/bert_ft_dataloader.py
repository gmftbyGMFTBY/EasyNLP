from header import *
from .utils import *
from .util_func import *


class BERTFTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if len(utterances) <= 1:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

class BERTFTWithNegDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.data = []
        if self.args['mode'] == 'train':
            data, responses = read_text_data_with_neg_q_r_neg(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context)
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]

                ids = item['input_ids']
                tids = item['token_type_ids']
                ids = self._length_limit(ids)
                tids = self._length_limit(tids)
                self.data.append({
                    'label': [1] + [0] * 10, 
                    'text': [(context, res) for res in [response] + candidate]
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context)
                # we only need 10 candidates, pos:neg = 1:9
                # compatible with the douban, ecommerce, ubuntu-v1 corpus
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                item = self.vocab.batch_encode_plus([
                    [context, res] for res in [response] + candidates
                ])
                ids = item['input_ids']
                tids = item['token_type_ids']
                ids = self._length_limit(ids)
                tids = self._length_limit(tids)
                self.data.append({
                    'label': [1] + [0] * 9, 
                    'ids': ids, 
                    'tids': tids, 
                })

    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            label = bundle['label']
            texts = bundle['text']
            item = self.vocab.batch_encode_plus(texts)
            ids = [torch.LongTensor(self._length_limit(i)) for i in item['input_ids']]
            tids = [torch.LongTensor(self._length_limit(i)) for i in item['token_type_ids']]
            return ids, tids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, tids, bundle['label']

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
        if self.args['mode'] == 'train':
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
        else:
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, tids, mask, label = ids.cuda(), tids.cuda(), mask.cuda(), label.cuda()
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask, 
            'label': label
        }

        
class BERTFTFullDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_full_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        
        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_ft(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
            for label, utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

            
class BERTTLFTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

            
class BERTFTIBNSDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.gray_cand_num = args['gray_cand_num']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_more_ng_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                self.data.append({
                    'ids': context,
                    'rids': response,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            rids = deepcopy(bundle['rids'])
            nrids = [deepcopy(item['rids']) for item in random.sample(self.data, self.gray_cand_num)]
            ids, tids = [], []
            for r in [rids] + nrids:
                cids = deepcopy(bundle['ids'])
                truncate_pair(cids, r, self.args['max_len'])
                ids.append([self.cls] + cids + [self.sep] + r + [self.sep])
                tids.append([0] * (len(cids) + 2) + [1] * (len(r) + 1))

            ids = [torch.LongTensor(i) for i in ids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids = [], []
            for i in batch:
                ids.extend(i[0])
                tids.extend(i[1])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, tids, mask = to_cuda(ids, tids, mask)
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
                'mask': mask, 
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }

class BERTFTFullDailyDialogDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_ft(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
            for label, utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }


class BERTFTRAGDataset(Dataset):

    '''RAG searcher is used'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])

        
        
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if len(utterances) <= 1:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.eos])
                context.pop()
                response = item[-1]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'tids': tids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }



class BERTFTRAGDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        
        data = []
        new_path = path.replace('train.txt', 'train_rag_bert_ft.txt')
        with open(new_path) as f:
            for line in tqdm(f.readlines()):
                data.append(json.loads(line))

        dataset = {}
        path = path.replace('train.txt', 'train_cross_domain.txt')
        with open(path) as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                dataset[line[0]] = line[1:]

        assert len(data) == len(dataset)

        self.data = []
        if self.args['mode'] == 'train':
            for index in tqdm(data):
                base_texts = dataset[index['conversation']]
                item = self.vocab.batch_encode_plus(base_texts, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-2]:
                    context.extend(u + [self.eos])
                context.pop()
                positive_response, negative_response = item[-2], item[-1]
                positive_context, netative_context = context, deepcopy(context)
                truncate_pair(positive_context, positive_response, self.args['max_len'])
                truncate_pair(negative_context, negative_response, self.args['max_len'])
                positive_ids = [self.cls] + positive_context + [self.sep] + positive_response + [self.sep]
                positive_tids = [0] * (len(positive_context) + 2) + [1] * (len(positive_response) + 1)
                negative_ids = [self.cls] + negative_context + [self.sep] + negative_response + [self.sep]
                negative_tids = [0] * (len(negative_context) + 2) + [1] * (len(negative_response) + 1)


                rag_text = [dataset[i][:-1] for i in index['gray_conversation']]
                contexts = []
                for rag_text_ in rag_text:
                    items = self.vocab.batch_encode_plus(rag_text_, add_special_tokens=False)['input_ids']
                    context = []
                    for u in item[:-1]:
                        context.extend(u + [self.eos])
                    context.pop()
                    response = item[-1]
                    truncate_pair(context, response, self.args['max_len'])
                    ids = [self.cls] + context + [self.sep] + response + [self.sep]
                    tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                    contexts.append((ids, tids))
                self.data.append({
                    'label': 1, 
                    'ids': positive_ids,
                    'tids': positive_tids,
                    'rag': contexts,
                })
                self.data.append({
                    'label': 0, 
                    'ids': negative_ids,
                    'tids': negative_tids,
                    'rag': contexts,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, tids = [], []
                context, responses = [], []
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
            context = bundle['context']
            responses = bundle['responses']
            return ids, tids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
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
                'mask': mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'tids': tids, 
                'mask': mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }


