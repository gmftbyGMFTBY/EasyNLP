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

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_bert_mask_da_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances_full(path, lang=self.args['lang'])
        self.data = []
        counter = 0
        for label, utterances in tqdm(data):
            item = self.vocab.encode(utterances[-1], add_special_tokens=False)
            ids = [self.cls] + item[:self.args['res_max_len']-2] + [self.sep]
            self.data.append({
                'ids': ids,
                'response': utterances[-1],
                'context': utterances[:-1],
                'index': counter
            })
            counter += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = mask_sentence(bundle['ids'], self.args['min_mask_num'], self.args['max_mask_num'], self.args['masked_lm_prob'], mask=self.mask, vocab_size=len(self.vocab), special_tokens=[self.mask, self.cls, self.sep, self.unk])
        ids = torch.LongTensor(bundle['ids'])
        return ids, bundle['context'], bundle['response'], bundle['index']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        context = [i[1] for i in batch]
        response = [i[2] for i in batch]
        index = [i[3] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, mask = to_cuda(ids, mask)
        return {
            'ids': ids, 
            'context': context,
            'response': response,
            'mask': mask, 
            'index': index,
        }
