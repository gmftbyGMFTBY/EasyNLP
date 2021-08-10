from header import *
from .utils import *
from .util_func import *


class SimCSEDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_simcse_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        data = list(chain(*[u for label, u in data if label == 1]))
        # ext_data = read_extended_douban_corpus(f'{args["root_dir"]}/data/ext_douban/train.txt')
        # data += ext_data
        data = list(set(data))
        print(f'[!] collect {len(data)} samples for simcse')

        self.data = []
        for idx in tqdm(range(0, len(data), 256)):
            utterances = data[idx:idx+256]
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = [[self.cls] + i[:self.args["res_max_len"]-2] + [self.sep] for i in item]
            self.data.extend(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = torch.LongTensor(self.data[i])
        return ids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }


class BERTSimCSEInferenceDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_simcse_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
        dataset = read_extended_douban_corpus(ext_path)
        self.data = []
        for utterance in tqdm(dataset):
            rids = length_limit_res(self.vocab.encode(utterance), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': utterance,
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text,
        }

        
class BERTSimCSEInferenceContextDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_simcse_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
        dataset = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=5)
        self.data = []
        counter = 0
        for label, utterances in tqdm(dataset):
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = [[self.cls] + i[:self.args['max_len']] + [self.sep] for i in item]
            self.data.append({
                'ids': ids, 
                'text': utterances,
                'index': counter
            })
            counter += 1
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = [torch.LongTensor(i) for i in bundle['ids']]
        utterances = bundle['text']
        return ids, utterances, bundle['index']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, text, index = [], [], []
        for i, j, k in batch:
            ids.extend(i)
            text.extend(j)
            index.extend([k] * len(i))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'text': text,
            'index': index,
        }
