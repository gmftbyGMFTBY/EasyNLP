from header import *
from .utils import *
from .util_func import *


class BERTMaskAugmentationDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.special_tokens = [self.pad, self.sep, self.cls]

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_bert_mask_da_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances_full(path, lang=self.args['lang'])
        self.data = []
        for label, utterances in tqdm(data):
            item = self.vocab.encode(utterances[-1], add_special_tokens=False)
            item = item[:self.args['res_max_len']-2]
            num_valid = len([i for i in item if i not in self.special_tokens])
            if num_valid < self.args['min_len']:
                continue
            ids = [self.cls] + item[:self.args['res_max_len']-2] + [self.sep]
            self.data.append({
                'ids': ids,
                'response': utterances[-1],
                'context': utterances[:-1],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['ids'], bundle['context'], bundle['response']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        context = [i[1] for i in batch]
        response = [i[2] for i in batch]
        return {
            'ids': ids, 
            'context': context,
            'response': response,
            'full': False,
        }

        
class BERTMaskAugmentationFullDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.special_tokens = [self.pad, self.sep, self.cls]

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_bert_mask_da_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances_full(path, lang=self.args['lang'])
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            rids = []
            for i in item: 
                i = i[:self.args['res_max_len']-2]
                num_valid = len([ii for ii in i if ii not in self.special_tokens])
                if num_valid < self.args['min_len']:
                    continue
                ids = [self.cls] + i[:self.args['res_max_len']-2] + [self.sep]
                rids.append(ids)
            if rids:
                self.data.append({
                    'ids': rids,
                    'response': utterances[-1],
                    'context': utterances[:-1],
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['ids'], bundle['context'], bundle['response']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, length = [], []
        for i in batch:
            ids.extend(i[0])
            length.append(len(i[0]))
        context = [i[1] for i in batch]
        response = [i[2] for i in batch]
        return {
            'ids': ids, 
            'context': context,
            'response': response,
            'length': length,
            'full': True,
        }
