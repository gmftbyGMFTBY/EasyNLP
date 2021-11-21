from header import *
from .utils import *
from .util_func import *


class BERTFTSCMDataset(Dataset):
    
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
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_scm_{suffix}.pt'
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
                    'label': label, 
                    'cids': context,
                    'rids': response,
                    'cands': item[:-1],
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
                })    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['cids'], bundle['rids']
            if len(bundle['cands']) < self.args['gray_cand_num_hn']:
                hn = bundle['cands'] + [i['rids'] for i in random.sample(self.data, self.args['gray_cand_num_hn'] - len(bundle['cands']))]
            else:
                hn = random.sample(bundle['cands'], self.args['gray_cand_num_hn'])
            en = [i['rids'] for i in random.sample(self.data, self.args['gray_cand_num_en'])]
            rids = [rids] + hn + en
            ids, tids = [], []
            for r in rids:
                c, r = deepcopy(cids), deepcopy(r)
                truncate_pair(c, r, self.args['max_len'])
                ids.append([self.cls] + c + [self.sep] + r + [self.sep])
                tids.append([0] * (len(c) + 2) + [1] * (len(r) + 1))
            ids = [torch.LongTensor(i) for i in ids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, tids
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            return ids, tids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids = [], []
            for a, b in batch:
                ids.extend(a)
                tids.extend(b)
        else:
            # batch size is batch_size * 10
            ids, tids, label = [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                label.extend(b[2])
            label = torch.LongTensor(label).cuda()
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
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
            }
