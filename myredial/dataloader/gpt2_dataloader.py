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
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            data = random.sample(data, 1000)
            self.data = []
            for text in tqdm(data):
                item = self.vocab.encode(text, add_special_tokens=False)
                for idx in range(0, len(item), self.args['max_len']-2):
                    ids = item[idx:idx+self.args['max_len']-2]
                    if len(ids) < self.args['min_len']:
                        continue
                    ids = [self.cls] + ids + [self.sep]
                    self.data.append({'ids': ids})
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
                    ids = [self.cls] + item[-(self.args['max_len']-1):]
                    item = self.vocab.encode(context+pos, add_special_tokens=False)
                    pos_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    item = self.vocab.encode(context+neg, add_special_tokens=False)
                    neg_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
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
            return ids
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
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return {
                'ids': ids, 
                'mask': mask, 
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

    '''Three learning objectives: NLL for GPT2, NLL for BERT, CLS for Positive and Negative training samples'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.neg_size = args['neg_size']
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_unlikelyhood_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_unlikelyhood(path)
            data = random.sample(data, 1000)
            self.data = []
            for utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids, cands, counter = [], [], 0
                for utterance in item:
                    if counter + len(utterance) + 3 > self.args['max_len']:
                        ids = list(chain(*ids))
                        ids = [self.cls] + ids + [self.sep] + utterance + [self.sep]
                        self.data.append({
                            'cids': ids,
                            'pos_rids': utterance,
                            'cands': cands,
                        })
                        ids, cands = [], []
                    else:
                        ids.append(utterance)
                        cands.append(utterance[:self.args['res_max_len']-2])
                        counter += len(utterance)
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
                    ids = [self.cls] + item[-(self.args['max_len']-1):]
                    item = self.vocab.encode(context+pos, add_special_tokens=False)
                    pos_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    item = self.vocab.encode(context+neg, add_special_tokens=False)
                    neg_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
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
            cands = torch.LongTensor(random.choice(bundle['cands']))
            return ids, cands
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
            ids = [i[0] for i in batch]
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return {
                'ids': ids, 
                'mask': mask, 
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
