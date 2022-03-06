from header import *
from .utils import *
from .util_func import *


class GPT2Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        random.seed(args['seed'])

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            self.data = []
            for text in tqdm(data):
                ids = self.vocab.encode(text, add_special_tokens=False)
                # do not add the [SEP] token, hurt the LM
                ids = [self.cls] + ids[-self.args['max_len']+1:]
                self.data.append({'ids': ids})
        else:
            data = read_text_data_line_by_line_sep(path)
            # for debug
            data = data[:100]
            self.data = []
            for ctx, res in tqdm(data):
                cids = self.vocab.encode(ctx, add_special_tokens=False)
                cids = [self.cls] + cids[-self.args['gen_max_ctx_len']+1:]
                rids = self.vocab.encode(res, add_special_tokens=False)
                # do not add the [SEP] token, hurt the LM
                rids = rids[:self.args['gen_max_len']]
                ids = cids + rids
                ids_label = [0] * len(cids) + rids
                self.data.append({
                    'ids': ids,
                    'cids': cids,
                    'ids_label': ids_label,
                    'response': res,
                    'context': ctx,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            return ids
        else:
            ids = torch.LongTensor(bundle['ids'])
            ids_label = torch.LongTensor(bundle['ids_label'])
            cids = torch.LongTensor(bundle['cids'])
            return ids, ids_label, cids, bundle['response'], bundle['context']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return {
                'ids': ids, 
                'mask': mask, 
            }
        else:
            ids = [i[0] for i in batch]
            ids_label = [i[1] for i in batch]
            cids = [i[2] for i in batch]
            response, context = batch[0][3], batch[0][4] 
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_label = pad_sequence(ids_label, batch_first=True, padding_value=self.pad)
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            cids_mask = generate_mask(cids)
            ids, ids_label, ids_mask, cids, cids_mask = to_cuda(ids, ids_label, ids_mask, cids, cids_mask)
            return {
                'ids': ids,
                'ids_label': ids_label,
                'ids_mask': ids_mask,
                'cids': cids,
                'cids_mask': cids_mask,
                'response': response,
                'context': context,
            }

            
            
class GPT2WithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_with_neg_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        random.seed(args['seed'])

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            self.data = []
            for text in tqdm(data):
                item = self.vocab.encode(text, add_special_tokens=False)
                if len(item) < self.args['min_train_len']:
                    continue
                for idx in range(0, len(item), self.args['max_len']):
                    ids = item[idx:idx+self.args['max_len']]
                    if len(ids) < self.args['min_len']:
                        continue

                    # negative samples: 16 tokens in negative responses
                    min_neg_len, max_neg_len = self.args['min_neg_len'], self.args['max_neg_len']
                    neg_len = random.randint(min_neg_len, max_neg_len)
                    index_range = list(range(0, len(item) - neg_len))
                    index_begin = random.choice(index_range)
                    negative_response = item[index_begin:index_begin+neg_len]
                    ctx_ids = ids[:-neg_len]
                    neg_ids = ctx_ids + negative_response 
                    neg_label = [0] * len(ctx_ids) + negative_response
                    
                    self.data.append({'ids': ids, 'neg_ids': neg_ids, 'neg_label': neg_label})
        else:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_simcse.pt'
            data = torch.load(path)
            # random sample 100 samples
            data = random.sample(data, 10)
            self.data = []
            for item in tqdm(data):
                context, pos, neg_responses = item['context'], item['pos_response'], item['neg_responses']
                for neg in neg_responses:
                    # prefix
                    item = self.vocab.encode(context, add_special_tokens=False)
                    ids = item[-self.args['max_len']:]
                    item = self.vocab.encode(context+pos, add_special_tokens=False)
                    pos_ids = item[:self.args['max_len']]
                    item = self.vocab.encode(context+neg, add_special_tokens=False)
                    neg_ids = item[:self.args['max_len']]
                    self.data.append({
                        'ids': ids,
                        'pos_ids': pos_ids,
                        'pos_text': context+pos,
                        'neg_ids': neg_ids,
                        'neg_text': context+neg,
                        'text': context,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            neg_ids = torch.LongTensor(bundle['neg_ids'])
            neg_label = torch.LongTensor(bundle['neg_label'])
            return ids, neg_ids, neg_label
        else:
            ids = torch.LongTensor(bundle['ids'])
            pos_ids = torch.LongTensor(bundle['pos_ids'])
            neg_ids = torch.LongTensor(bundle['neg_ids'])
            return ids, pos_ids, neg_ids, bundle['pos_text'], bundle['neg_text'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, neg_ids = [i[0] for i in batch], [i[1] for i in batch]
            neg_label = [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
            neg_label = pad_sequence(neg_label, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            neg_mask = generate_mask(neg_ids)
            ids, mask = to_cuda(ids, mask)
            neg_ids, neg_mask, neg_label = to_cuda(neg_ids, neg_mask, neg_label)
            return {
                'ids': ids, 
                'mask': mask, 
                'neg_ids': neg_ids,
                'neg_mask': neg_mask,
                'neg_label': neg_label,
            }
        else:
            ids = [i[0] for i in batch]
            pos_ids = [i[1] for i in batch]
            neg_ids = [i[2] for i in batch]
            pos_text = [i[3] for i in batch]
            neg_text = [i[4] for i in batch]
            text = [i[5] for i in batch]

            # pad from the left side, batch first
            max_length = max([len(i) for i in ids])
            n_ids = []
            for i in ids:
                ids_ = torch.cat([torch.LongTensor([self.pad] * (max_length - len(i))), i])
                n_ids.append(ids_)
            ids = torch.stack(n_ids)
            mask = generate_mask(ids)
            
            pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad)
            pos_ids_mask = generate_mask(pos_ids)
            neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
            neg_ids_mask = generate_mask(neg_ids)
            ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask = to_cuda(ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask)
            return {
                'ids': ids, 
                'mask': mask, 
                'pos_ids': pos_ids, 
                'pos_ids_mask': pos_ids_mask, 
                'neg_ids': neg_ids, 
                'neg_ids_mask': neg_ids_mask, 
                'pos_text': pos_text,
                'text': text,
                'neg_text': neg_text,
            }

            
class GPT2UnlikelyhoodDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        random.seed(args['seed'])

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            responses = list(set(data))
            self.data = []
            for text in tqdm(data):
                ids = self.vocab.encode(text, add_special_tokens=False)
                ids = [self.cls] + ids[-self.args['max_len']+1:]
                self.data.append({'ids': ids})
        else:
            data = read_text_data_line_by_line_sep(path)
            # for debug
            data = data[:100]
            self.data = []
            for ctx, res in tqdm(data):
                cids = self.vocab.encode(ctx, add_special_tokens=False)
                cids = [self.cls] + cids[-self.args['gen_max_ctx_len']+1:]
                rids = self.vocab.encode(res, add_special_tokens=False)
                # do not add the [SEP] token, hurt the LM
                rids = rids[:self.args['gen_max_len']]
                ids = cids + rids
                ids_label = [0] * len(cids) + rids
                self.data.append({
                    'ids': ids,
                    'cids': cids,
                    'ids_label': ids_label,
                    'response': res,
                    'context': ctx,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            gt_ids = torch.LongTensor(bundle['ids'])
            # noise input
            if random.random() > 0.5:
                ctx = bundle['ids'][:-self.args['res_len']]
                res = random.choice(self.data)['ids'][:self.args['res_len']]
                noise_ids = ctx + res
                noise_ids = torch.LongTensor(noise_ids)
                cls_label = 0
            else:
                noise_ids = torch.LongTensor(bundle['ids'])
                cls_label = 1
            return gt_ids, noise_ids, cls_label, len(noise_ids) - 1
        else:
            ids = torch.LongTensor(bundle['ids'])
            ids_label = torch.LongTensor(bundle['ids_label'])
            cids = torch.LongTensor(bundle['cids'])
            return ids, ids_label, cids, bundle['response'], bundle['context']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            gpt2_ids = [i[0] for i in batch]
            noise_ids = [i[1] for i in batch]
            cls_label = torch.LongTensor([i[2] for i in batch])
            last_token_pos = [i[3] for i in batch]
            gpt2_ids = pad_sequence(gpt2_ids, batch_first=True, padding_value=self.pad)
            noise_ids = pad_sequence(noise_ids, batch_first=True, padding_value=self.pad)
            gpt2_mask = generate_mask(gpt2_ids)
            noise_mask = generate_mask(noise_ids)
            gpt2_ids, gpt2_mask, noise_ids, noise_mask, cls_label = to_cuda(gpt2_ids, gpt2_mask, noise_ids, noise_mask, cls_label)
            return {
                'gpt2_ids': gpt2_ids,
                'gpt2_mask': gpt2_mask,
                'noise_ids': noise_ids,
                'noise_mask': noise_mask,
                'cls_label': cls_label,
                'last_token_pos': last_token_pos,
            }
        else:
            ids = [i[0] for i in batch]
            ids_label = [i[1] for i in batch]
            cids = [i[2] for i in batch]
            response, context = batch[0][3], batch[0][4] 
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_label = pad_sequence(ids_label, batch_first=True, padding_value=self.pad)
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            cids_mask = generate_mask(cids)
            ids, ids_label, ids_mask, cids, cids_mask = to_cuda(ids, ids_label, ids_mask, cids, cids_mask)
            return {
                'ids': ids,
                'ids_label': ids_label,
                'ids_mask': ids_mask,
                'cids': cids,
                'cids_mask': cids_mask,
                'response': response,
                'context': context,
            }

            
class GPT2InferenceDataset(Dataset):

    '''only for gpt2 inference dataset and writer-gen-test dataset'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')

        path = f'{os.path.split(path)[0]}/{args["file_name"]}.txt'
        with open(path) as f:
            data = []
            for line in f.readlines():
                line = json.loads(line.strip())
                data.append(line['prefix'])
        print(f'[!] load {len(data)} samples for inference from {path}')
        
        self.data = []
        for ctx in tqdm(data):
            ids = self.vocab.tokenize(ctx)
            ids = self.vocab.convert_tokens_to_ids(ids)
            ids = ids[-self.args['gen_max_ctx_len']+1:]
            text = ''.join(self.vocab.convert_ids_to_tokens(ids))
            self.data.append({'ids': ids, 'text': text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['ids'], bundle['text']

    def save(self):
        pass
        
    def collate(self, batch):
        max_length = max([len(i[0]) for i in batch])
        tokens = [i[0] for i in batch]
        texts = [i[1] for i in batch]
        ids = torch.LongTensor([[self.pad] * (max_length - len(i)) + i for i in tokens])
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
        ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
        return {
           'ids': ids,
           'pos_ids': pos_ids,
           'ids_mask': ids_mask,
           'text': texts,
        }
