from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class DocTTTTTQueryDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_doctttttquery_{suffix}.pt'
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
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                cids = ids[:self.args['max_len']]    # ignore [CLS] and [SEP]
                rids = rids[-self.args['res_max_len']:]
                ids = rids + [self.sep] + cids
                label = [self.pad] * len(rids) + [self.pad] + cids
                self.data.append({
                    'ids': ids,
                    'label': label
                })
        else:
            # for inference 
            path = path.replace('test', 'train')
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = item[-1][-self.args['res_max_len']:]
                self.data.append({
                    'ids': ids,
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            label = torch.LongTensor(bundle['label'])
            return ids, label
        else:
            return bundle['ids']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, label = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            label = pad_sequence(label, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, ids_mask, label = to_cuda(ids, ids_mask, label)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
                'label': label,
            }
        else:
            # batch size is batch_size * 10
            max_length = max([len(i) for i in batch])
            ids = [[self.pad] * (max_length -len(item)) + item + [self.sep] for item in batch]
            ids = torch.LongTensor(ids)
            ids_mask = generate_mask(ids)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            return {
                'ids': ids, 
                'pos_ids': pos_ids,
                'ids_mask': ids_mask,
            }
