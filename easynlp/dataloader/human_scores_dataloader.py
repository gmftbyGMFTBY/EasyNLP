from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class HumanScoresDataset(Dataset):

    '''
    1. RRS ranking test set
    2. DailyDialog GRADE dataset'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.data = []
        with open(path) as f:
            data = []
            for line in f.readlines():
                item = json.loads(line.strip())
                data.append(item)

            data = data[:100]

            for item in tqdm(data):
                score = item['score']
                context = [i.strip() for i in item['context'].split('|||') if i.strip()]
                response = item['response']
                ground_truth = item['ground_truth']
                item = self.vocab.batch_encode_plus(context, add_special_tokens=False)['input_ids']
                ids = []
                for u in item:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                item = self.vocab.batch_encode_plus([response, ground_truth], add_special_tokens=False)['input_ids']
                rids = item[0][:(self.args['res_max_len']-2)]
                ground_truth = item[1][:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                ground_truth = [self.cls] + ground_truth + [self.sep]
                self.data.append({
                    'score': score,
                    'ids': ids,
                    'rids': rids,
                    'ground_truth': ground_truth
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = torch.LongTensor(bundle['rids'])
        return ids, rids, bundle['score']

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        rids = [i[1] for i in batch]
        scores = [i[2] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids, pad_token_idx=self.pad)
        ids, ids_mask, rids, rids_mask = to_cuda(ids, ids_mask, rids, rids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'score': scores
        }

class HumanScoresInteractionDataset(Dataset):

    '''
    1. RRS ranking test set
    2. DailyDialog GRADE dataset'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.data = []
        with open(path) as f:
            data = []
            for line in f.readlines():
                item = json.loads(line.strip())
                data.append(item)

            for item in tqdm(data):
                score = item['score']
                context = [i.strip() for i in item['context'].split('|||') if i.strip()]
                response = item['response']
                ground_truth = item['ground_truth']
                item = self.vocab.batch_encode_plus(context, add_special_tokens=False)['input_ids']
                context = []
                for u in item:
                    context.extend(u + [self.eos])
                context.pop()
                item = self.vocab.batch_encode_plus([response, ground_truth], add_special_tokens=False)['input_ids']
                response = item[0]
                truncate_pair(context, response, self.args['max_len'])
                ids = [self.cls] + context + [self.sep] + response + [self.sep]
                tids = [0] * (len(context) + 2) + [1] * (len(response ) + 1)
                self.data.append({
                    'score': score,
                    'ids': ids,
                    'tids': tids,
                    'ground_truth': ground_truth
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        tids = torch.LongTensor(bundle['tids'])
        return ids, tids, bundle['score']

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        tids = [i[1] for i in batch]
        scores = [i[2] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
        return {
            'ids': ids, 
            'tids': tids,
            'mask': ids_mask, 
            'score': scores
        }


class HumanScoresTextualDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.data = []
        with open(path) as f:
            data = []
            for line in f.readlines():
                item = json.loads(line.strip())
                data.append(item)

            for item in tqdm(data):
                score = item['score']
                response = item['response']
                ground_truth = item['ground_truth']
                self.data.append({
                    'score': score,
                    'response': response,
                    'ground_truth': ground_truth
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['response'], bundle['ground_truth'], bundle['score']

    def save(self):
        pass
        
    def collate(self, batch):
        response = [i[0] for i in batch]
        reference = [i[1] for i in batch]
        scores = [i[2] for i in batch]
        return {
            'response': response,
            'reference': reference,
            'score': scores
        }


