from header import *
from .utils import *
from .util_func import *


class BARTFTDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_bart_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if len(utterances) <= 1:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                context = []
                for u in item[:-1]:
                    context.extend(u + [self.sep])
                context.pop()
                response = item[-1]
                ids = [self.cls] + context[-self.args['max_len']+2:] + [self.sep]
                rids = [self.cls] + response[:self.args['res_max_len']-2] + [self.sep]
                self.data.append({
                    'label': label, 
                    'ids': ids,
                    'rids': rids,
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids, rids = [], []
                context, responses = [], []
                for b in batch:
                    label = b[0]
                    utterances = b[1]
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids = []
                    for u in item[:-1]:
                        cids.extend(u + [self.sep])
                    cids.pop()
                    rids_ = item[-1]
                    ids_ = [self.cls] + cids[-self.args['max_len']+2:] + [self.sep]
                    rids_ = [self.cls] + rids_[:self.args['res_max_len']-2] + [self.sep]
                    ids.append(ids_)
                    rids.append(rids_)
                    responses.append(utterances[-1])
                context = ' [SEP] '.join(utterances[:-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            label = bundle['label']
            return ids, rids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            context = bundle['context']
            responses = bundle['responses']
            return ids, rids, bundle['label'], context, responses

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, context, responses = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        rids_mask = generate_mask(rids)
        label = torch.LongTensor(label)
        ids, rids, ids_mask, rids_mask, label = to_cuda(ids, rids, ids_mask, rids_mask, label)
        if self.args['mode'] == 'train':
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask, 
                'label': label
            }
        else:
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask, 
                'label': label,
                'context': context,
                'responses': responses,
            }
