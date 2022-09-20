from header import *
from .utils import *
from .util_func import *


class BERTFTInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        path = path.replace('train.txt', 'train_only_groundtruth.txt')
        with open(path) as f:
            data = [line.strip() for line in f.readlines()]
        
        self.data = []
        for idx, line in tqdm(enumerate(data)):
            utterances = line.split('\t')[1:]
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            context = []
            for u in item[:-1]:
                context.extend(u + [self.eos])
            context.pop()
            response = item[-1]
            truncate_pair(context, response, self.args['max_len'])
            ids = [self.cls] + context + [self.sep] + response + [self.sep]
            tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
            self.data.append({
                'ids': ids,
                'tids': tids,
                'text': idx
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        tids = torch.LongTensor(bundle['tids'])
        return ids, tids, bundle['text']

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, text = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask, 
            'text': text
        }
