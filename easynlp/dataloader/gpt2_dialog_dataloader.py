from header import *
from .utils import *
from .util_func import *


class GPT2DialogDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_dialog_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        data = read_text_data_line_by_line(path)
        self.data = []
        for text in tqdm(data):
            items = text.strip().split('\t')
            if items[0] == '0':
                continue
            items = items[1:]
            context, response = items[:-1], items[-1]
            context = f' [SEP] '.join(context)
            context = self.vocab.encode(context, add_special_tokens=False)
            response = self.vocab.encode(response, add_special_tokens=False)
            truncate_pair(context, response, self.args['max_len'])
            tokens = context + [self.sep] + response
            labels = [self.pad] * (len(context) + 1) + response
            self.data.append({
                'ids': tokens, 
                'context': f' [SEP] '.join(items[:-1]),
                'response': items[-1],
                'label': labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]['ids'], self.data[i]['context'], self.data[i]['response'], self.data[i]['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        # if self.args['mode'] == 'train':
        ids = [torch.LongTensor(i[0]) for i in batch]
        label = [torch.LongTensor(i[-1]) for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_label = pad_sequence(label, batch_first=True, padding_value=self.pad)
        ids, ids_label = ids[:, :-1], ids_label[:, 1:]
        mask = generate_mask(ids)
        ids, mask, ids_label = to_cuda(ids, mask, ids_label)
        return {
            'ids': ids, 
            'ids_mask': mask, 
            'ids_label': ids_label,
            'pos_ids': None,
        }
        # else:
        #     context, response = [i[1] for i in batch], [i[2] for i in batch]
        #     ids = [[i[0]] * self.args['inference_num'] for i in batch]
        #     ids = [torch.LongTensor(i) for i in list(chain(*ids))]
        #     ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        #     ids, ids_label = ids[:-1], ids[1:]
        #     ids_mask = generate_mask(ids)
        #     pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
        #     ids, ids_mask, pos_ids, ids_label = to_cuda(ids, ids_mask, pos_ids, ids_label)
        #     return {
        #         'ids': ids,
        #         'pos_ids': pos_ids,
        #         'ids_mask': ids_mask,
        #         'context': context,
        #         'response': response,
        #     }
