from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualFullHierDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_hier_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            # data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                self.data.append({
                    'ids': ids[:-1],
                    'rids': ids[-1],
                    'turn_length': len(ids) - 1,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['mode'] == 'valid' and args['dataset'] in ['ubuntu']:
                data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    context_list = utterances[:-1]
                    ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                    rids.append(ids[-1])
                    if label == 1:
                        gt_text.append(utterances[-1])

                    # for dual-bert
                    ctext = ' [SEP] '.join(utterances[:-1])

                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids[:-1],
                    'rids': rids,
                    'text': gt_text,
                    'turn_length': len(ids) - 1,
                    'context_list': context_list, 
                    'ctext': ctext,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in bundle['ids'][-self.args['max_turn_length']:]]
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, len(ids)
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids'][-self.args['max_turn_length']:]]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], len(ids), bundle['context_list'], bundle['text'], bundle['ctext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = []
            for i in batch:
                ids.extend(i[0])
            rids = [i[1] for i in batch]
            turn_length = [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            if rids.size(-1) < self.args['res_max_len']:
                padding_matrix = torch.LongTensor([self.pad] * (self.args['res_max_len'] - rids.size(-1))).unsqueeze(0).expand(rids.size(0), -1)
                rids = torch.cat([rids, padding_matrix], dim=-1)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'turn_length': turn_length,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, turn_length, context_list, text, ctext = batch[0]
            turn_length = [turn_length]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            if ids.size(-1) < self.args['mv_num']:
                ids = torch.cat([ids, torch.zeros(ids.size(0), self.args['mv_num'] - ids.size(-1)).to(torch.long)], dim=-1)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, ids_mask, rids_mask, label = to_cuda(ids, rids, ids_mask, rids_mask, label)
            return {
                'ids': ids, 
                'ids_mask': ids_mask,
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'turn_length': turn_length,
                'context_list': context_list,
                'text': text,
                'ctext': ctext
            }

class BERTDualHierTrsDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hier_trs_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                self.data.append({
                    'ids': ids,
                    'turn_length': len(ids),
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['mode'] == 'valid' and args['dataset'] in ['ubuntu']:
                data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                    rids.append(ids[-1])
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids[:-1],
                    'rids': rids,
                    'text': gt_text,
                    'turn_length': len(ids) - 1
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            return ids, bundle['turn_length']
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['turn_length']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = []
            for i in batch:
                ids.extend(i[0])
            turn_length = [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, ids_mask= to_cuda(ids, ids_mask)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'turn_length': turn_length,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, turn_length = batch[0]
            turn_length = [turn_length]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'turn_length': turn_length
            }


class BERTDualBM25SCMLiteDistDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')


        self.data = []
        if self.args['mode'] == 'train':
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray_for_scm.rar'
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
            suffix = args['tokenizer'].replace('/', '_')
            self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_hier_dist_{suffix}.pt'
            if os.path.exists(self.pp_path):
                self.data = torch.load(self.pp_path)
                print(f'[!] load preprocessed file from {self.pp_path}')
                return None

            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['mode'] == 'valid' and args['dataset'] in ['ubuntu']:
                data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                    rids.append(ids[-1])
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids[:-1],
                    'rids': rids,
                    'text': gt_text,
                    'turn_length': len(ids) - 1
                })    
                
    def __len__(self):
        if self.args['mode'] == 'train':
            return self.size
        else:
            return len(self.data)

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            # dr-bert-v2
            line = self.reader.get_line(i)
            item = json.loads(line.strip())
            ctx, res = item['q'], item['r']
            # cands = item['q_q_nr'] + item['single_nr']
            # cands = random.sample(cands, self.args['gray_cand_num'])
            # utterances = ctx + [res] + cands
            utterances = ctx + [res]
            ids_ = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids_]

            ids = [torch.LongTensor(i) for i in ids[-self.args['max_turn_length']:]]
            # cids_, rids_ = ids[:len(ctx)], ids[len(ctx):]
            cids_, rids_ = ids[:-1], ids[-1]
            turn_length = len(ctx)

            # dr-bert
            cids, rids = ids_[:-1], ids_[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['dr_bert_max_len']-2):]
            ids = [self.cls] + ids + [self.sep]

            rids = [self.cls] + rids[:self.args['dr_bert_res_max_len']-2] + [self.sep] 
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            return cids_, rids_, turn_length, ids, rids
        else:
            bundle = self.data[i]
            ids = [torch.LongTensor(i) for i in bundle['ids'][-self.args['max_turn_length']:]]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], len(ids)

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids_ = []
            for i in batch:
                ids_.extend(i[0])
            rids_ = [i[1] for i in batch]
            ids = [i[3] for i in batch]
            rids = [i[4] for i in batch]
            turn_length = [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_ = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
            rids_ = pad_sequence(rids_, batch_first=True, padding_value=self.pad)
            ids_mask_ = generate_mask(ids_)
            rids_mask_ = generate_mask(rids_)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids_, rids_, ids_mask_, rids_mask_ = to_cuda(ids_, rids_, ids_mask_, rids_mask_)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ids_': ids_, 
                'rids_': rids_, 
                'ids_mask_': ids_mask_, 
                'rids_mask_': rids_mask_,
                'turn_length': turn_length,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, turn_length = batch[0]
            turn_length = [turn_length]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'turn_length': turn_length
            }


class BERTFTFullHierDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_full_hier_{suffix}.pt'
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
                ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                self.data.append({
                    'ids': ids,
                    'turn_length': len(ids),
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['mode'] == 'valid' and args['dataset'] in ['ubuntu']:
                data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                    if label == 1:
                        gt_text.append(utterances[-1])
                    rids.append(ids)

                    # for dual-bert
                    ctext = ' [SEP] '.join(utterances[:-1])
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    ids = [self.cls] + ids + [self.sep]

                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': rids,
                    'text': gt_text,
                    'turn_length': len(ids),
                    'ctext': ctext,
                    'dual_bert_ids': ids
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in bundle['ids'][-self.args['max_turn_length']:]]
            return ids, len(ids) 
        else:
            ids = [[torch.LongTensor(i) for i in item[-self.args['max_turn_length']:]] for item in bundle['ids']]
            turn_length = [len(item) for item in ids]
            return ids, bundle['label'], turn_length, bundle['text'], bundle['ctext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = []
            for i in batch:
                ids.extend(i[0])
            turn_length = [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, ids_mask = to_cuda(ids, ids_mask)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'turn_length': turn_length,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, label, turn_length, text, ctext = batch[0]
            ids = list(chain(*ids))
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            label = torch.LongTensor(label)
            ids, ids_mask, label = to_cuda(ids, ids_mask, label)
            ipdb.set_trace()
            return {
                'ids': ids, 
                'ids_mask': ids_mask,
                'label': label,
                'turn_length': turn_length,
                'text': text,
                'ctext': ctext,
            }
