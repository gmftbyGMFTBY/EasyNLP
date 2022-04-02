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

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_pretrained_model'])
        self.bert_cls = self.bert_tokenizer.convert_tokens_to_ids('[CLS]')
        self.bert_sep = self.bert_tokenizer.convert_tokens_to_ids('[SEP]')

        self.data = []
        data = read_text_data_line_by_line(path)
        self.data = []
        for text in tqdm(data):
            items = text.strip().split('\t')
            if items[0] == '0':
                continue
            items = items[1:]
            context_list, ground_truth = items[:-1], items[-1]
            context = ' [SEP] '.join(context_list)
            context = self.vocab.encode(context, add_special_tokens=False)
            response = self.vocab.encode(ground_truth, add_special_tokens=False)
            bert_response = self.bert_tokenizer.encode(response, add_special_tokens=False)
            bert_response = [self.bert_cls] + bert_response[:self.args['bert_res_len']] + [self.bert_sep]
            truncate_pair(context, response, self.args['max_len'])
            if self.args['mode'] == 'train':
                tokens = context + [self.sep] + response + [self.sep]
                start_index = len(context) + 1
                length = len(response)

                self.data.append({
                    'ids': tokens, 
                    'response': bert_response,
                    'start_index': start_index,
                    'length': length
                })
            else:
                tokens = context + [self.sep]
                start_index, length = 0, 0
                self.data.append({
                    'context_list': context_list,
                    'ground_truth': ground_truth,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            return torch.LongTensor(bundle['ids']), torch.LongTensor(bundle['response']), bundle['start_index'], bundle['length']
        else:
            return bundle['context_list'], bundle['ground_truth']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            bert_ids = [i[1] for i in batch]
            s = torch.LongTensor([i[2] for i in batch])
            l = torch.LongTensor([i[3] for i in batch])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            bert_ids = pad_sequence(bert_ids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            bert_mask = generate_mask(bert_ids)
            ids, mask, bert_ids, bert_mask, s, l = to_cuda(ids, mask, bert_ids, bert_mask, s, l)
            return {
                'ids': ids, 
                'ids_mask': mask, 
                'bert_ids': bert_ids,
                'bert_ids_mask': bert_mask,
                's': s,
                'l': l
            }
        else:
            context_list, ground_truth = batch[0]
            return {
                'context_list': context_list,
                'ground_truth': ground_truth,
            }
