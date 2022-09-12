from header import *
from itertools import islice
from .utils import *
from .util_func import *
from .randomaccess import *
from .check import *


class KNNLMInferenceDataset(Dataset):

    '''only for wikitext103 dataset'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        if self.args['dataset'] in ['wikitext103']:
            path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/base_data.txt'
            # self.pad = self.vocab.eos_token_id
            self.pad = self.vocab.pad_token_id
        elif self.args['dataset'] in ['copygeneration_lawmt']:
            path = f'/apdcephfs/share_916081/johntianlan/copygeneration_lawmt/base_data.txt'
            # self.pad = self.vocab.eos_token_id
            self.pad = self.vocab.pad_token_id
        elif self.args['dataset'] in ['en_wiki']:
            path = f'/apdcephfs/share_916081/johntianlan/copygeneration_en_wiki/base_data.txt'
            print(f'[!] prepare to load data from {path}')
            # self.pad = self.vocab.eos_token_id
            self.pad = self.vocab.pad_token_id
        else:
            path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/base_data.txt'
            self.pad = self.vocab.pad_token_id

        if self.args['dataset'] in ['wikitext103', 'copygeneration_lawmt', 'en_wiki']:
            self.data = []
            counter = 0
            with open(path) as f:
                for line in tqdm(f.readlines()):
                    items = line.strip().split('\t')
                    context = '\t'.join(items[:-1])
                    idx = items[-1]
                    tokens = self.vocab.encode(context, add_special_tokens=False)
                    # make the chunk (512)
                    for i in range(0, len(tokens), 512):
                        subsequence = tokens[i:i+512]
                        if len(subsequence) < 64:
                            # shorter context is useless
                            continue
                        self.data.append(subsequence)
                        counter += len(subsequence) - 1
                    if len(self.data) >= 1300000:
                        break
            print(f'[!] collect {len(self.data)} samples and {counter} key-values')
            self.size = len(self.data)
        else:
            self.data = []
            counter = 0
            with open(path) as f:
                for line in tqdm(f.readlines()):
                    item = json.loads(line)
                    text = item['results']
                    context = ' '.join(text)
                    tokens = self.vocab.encode(context, add_special_tokens=False)
                    self.data.append(tokens[:512])
                    counter += len(tokens[:512]) - 1
                    if len(self.data) >= 1000000:
                        break
            print(f'[!] collect {len(self.data)} samples and {counter} key-values')
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        ids = self.data[i]
        return torch.LongTensor(ids)

    def save(self):
        pass
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }
