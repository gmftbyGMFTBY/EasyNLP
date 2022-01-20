from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class TraditionalRSDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_traditional_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                item = [i[-self.args['max_uttr_len']:] for i in item[-self.args['max_uttr_num']:]]
                cids, rids = item[:-1], item[-1]
                self.data.append({
                    'ids': cids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if self.args['dataset'] == 'ubuntu':
                data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    item = [i[-self.args['max_uttr_len']:] for i in item[-self.args['max_uttr_num']:]]
                    cids, rids_ = item[:-1], item[-1]
                    rids.append(rids_)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': cids,
                    'rids': rids,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            random_neg_idx = random.sample(range(len(self.data)), self.args['neg_candi_num'])
            random_neg = [self.data[i]['rids'] for i in random_neg_idx]
            ids = [uttr + [self.pad] * (self.args['max_uttr_len'] - len(uttr)) for uttr in bundle['ids']]
            for _ in range(max(0, self.args['max_uttr_num'] - len(ids))):
                ids.append([self.pad for _ in range(self.args['max_uttr_len'])])
            rids = bundle['rids'] + [self.pad] * (self.args['max_uttr_len'] - len(bundle['rids']))
            neg_rids = [r + [self.pad] * (self.args['max_uttr_len'] - len(r)) for r in random_neg]
            return ids, [rids] + neg_rids, bundle['ctext'], bundle['rtext']
        else:
            ids = [uttr + [self.pad] * (self.args['max_uttr_len'] - len(uttr)) for uttr in bundle['ids']]
            for _ in range(max(0, self.args['max_uttr_num'] - len(ids))):
                ids.append([self.pad for _ in range(self.args['max_uttr_len'])])
            rids = [r + [self.pad] * (self.args['max_uttr_len'] - len(r)) for r in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            ids, rids = to_cuda(ids, rids)
            return {
                'ids': ids, 
                'rids': rids, 
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            label = [i[2] for i in batch]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            label = torch.LongTensor(label)
            ids, rids, label = to_cuda(ids, rids, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'label': label,
            }
