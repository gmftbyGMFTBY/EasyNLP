from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class ColBERTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_colbert_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            cache = []
            for label, utterances in tqdm(data):
                if label == 1 and cache:
                    assert len(cache) == 2
                    self.data.append({
                        'ids': cache[0][0] ,
                        'rids': cache[0][1],
                        'nrids': cache[1][1]
                    })
                    cache = []
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                cache.append((ids, rids))
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 50)):
                batch = data[i:i+50]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            nrids = torch.LongTensor(bundle['nrids'])
            return ids, rids, nrids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            nrids = [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            nrids = pad_sequence(nrids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            nrids_mask = generate_mask(nrids)
            ids, rids, nrids, ids_mask, rids_mask, nrids_mask = to_cuda(ids, rids, nrids, ids_mask, rids_mask, nrids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'nrids': nrids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'nrids_mask': nrids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class ColBERTV2Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_colbert_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            cache = []
            for label, utterances in tqdm(data):
                if label == 1 and cache:
                    assert len(cache) == 2
                    self.data.append({
                        'ids': cache[0][0] ,
                        'rids': cache[0][1],
                        'nrids': cache[1][1]
                    })
                    cache = []
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                cache.append((ids, rids))
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 1000)):
                batch = data[i:i+1000]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            bsz, ids_l = ids.size()
            _, rids_l = rids.size()

            # pad to the max length
            ids = torch.cat([
                    ids, 
                    torch.zeros(bsz, self.args['max_len'] - ids_l).to(torch.long)
                ], dim=-1
            )
            rids = torch.cat([
                    rids, 
                    torch.zeros(bsz, self.args['res_max_len'] - rids_l).to(torch.long)
                ], dim=-1
            )

            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

class ColBERTV2FullDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_colbert_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
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
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 50)):
                batch = data[i:i+50]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            bsz, ids_l = ids.size()
            _, rids_l = rids.size()

            # pad to the max length
            ids = torch.cat([
                    ids, 
                    torch.zeros(bsz, self.args['max_len'] - ids_l).to(torch.long)
                ], dim=-1
            )
            rids = torch.cat([
                    rids, 
                    torch.zeros(bsz, self.args['res_max_len'] - rids_l).to(torch.long)
                ], dim=-1
            )

            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

class ColBERTV2HNDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.data = []
        if self.args['mode'] == 'train':
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.rar'
            path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccessReader Object over')
            else:
                self.reader = RandomAccessReader(path)
                self.reader.init()
                torch.save(self.reader, rar_path)
            self.reader.init_file_handler()
            self.size = self.reader.size
            print(f'[!] dataset size: {self.size}')
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i)
            item = json.loads(line.strip())

            ctx, res = item['q'], item['r']
            cands = item['nr']
            ipdb.set_trace()
            cands = random.sample(cands, self.args['gray_cand_num'])
            utterances = ctx + [res] + cands
            tokens = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = tokens[:len(ctx)]
            rids = tokens[len(ctx)]
            crids = tokens[-len(cands):]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = [self.cls] + ids[-self.args['max_len']+2:] + [self.sep]
            rids = [self.cls] + rids[:self.args['res_max_len']-2] + [self.sep]
            crids = [[self.cls] + r[:self.args['res_max_len']-2] + [self.sep] for r in crids]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            crids = [torch.LongTensor(i) for i in crids]
            return ids, [rids] + crids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for _, r in batch:
                rids.extend(r)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

