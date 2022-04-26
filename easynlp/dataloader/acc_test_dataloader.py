from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualAccDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_acc_test_dual_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = path.replace('test.txt', 'test_acc.txt')
        data = read_text_data_utterances(path, lang=self.args['lang'])
        for i in tqdm(range(len(data))):
            try:
                label, utterances = data[i]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids_ = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids_[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'label': label,
                    'ids': ids,
                    'rids': rids,
                })    
            except:
                continue
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = torch.LongTensor(bundle['rids'])
        return ids, rids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, rids, labels = [], [], []
        for a, b, c in batch:
            ids.append(a)
            rids.append(b)
            labels.append(c)
        
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        rids_mask = generate_mask(rids)
        label = torch.LongTensor(labels)
        ids, ids_mask, rids, rids_mask, label = to_cuda(ids, ids_mask, rids, rids_mask, label)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'label': label,
        }


class BERTFTAccDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_acc_test_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = path.replace('test.txt', 'test_acc.txt')
        data = read_text_data_utterances(path, lang=self.args['lang'])
        for i in tqdm(range(len(data))):
            try:
                label, utterances = data[i]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = []
                for u in item[:-1]:
                    ids.extend(u + [self.eos])
                ids.pop()
                response = item[-1]
                context = ids
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
                self.data.append({
                    'label': label,
                    'ids': ids,
                    'tids': tids,
                })    
            except Exception as error:
                print(error)
                continue
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        tids = torch.LongTensor(bundle['tids'])
        return ids, tids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, tids, labels = [], [], []
        for a, b, c in batch:
            ids.append(a)
            tids.append(b)
            labels.append(c)
        
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        label = torch.LongTensor(labels)
        ids, ids_mask, tids, label = to_cuda(ids, ids_mask, tids, label)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'tids': tids, 
            'label': label,
        }
