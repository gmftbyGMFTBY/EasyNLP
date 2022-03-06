from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualWRDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_wr_full_{self.args["full_turn_length"]}_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            response = utterances[-1]
            kw = jieba.analyse.extract_tags(response)
            if len(kw) == 0:
                continue
            utterances = utterances[:-1]
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = []
            for u in item:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            ids = [self.cls] + ids + [self.sep]

            rids = self.vocab.batch_encode_plus(kw, add_special_tokens=False)['input_ids']
            rids = [[self.cls] + i + [self.sep] for i in rids]
            self.data.append({
                'ids': ids,
                'rids': rids,
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = torch.LongTensor(random.choice(bundle['rids']))
        return ids, rids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, rids = [i[0] for i in batch], [i[1] for i in batch]
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
