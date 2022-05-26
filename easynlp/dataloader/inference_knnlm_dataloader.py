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
        self.pad = self.vocab.eos_token_id
        path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/base_data.txt'
        self.pp_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/base_data_pp.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            self.size = len(self.data)
            print(f'[!] load preprocessed file from {self.pp_path}: {self.size}')
            return None

        self.data = []
        counter = 0
        with open(path) as f:
            for line in tqdm(f.readlines()):
                context, idx = line.strip().split('\t')
                tokens = self.vocab.encode(context, add_special_tokens=False)
                # make the chunk (512)
                for i in range(0, len(tokens), 512):
                    subsequence = tokens[i:i+512]
                    if len(subsequence) < 64:
                        # shorter context is useless
                        continue
                    self.data.append(subsequence)
                    counter += len(subsequence) - 1
        print(f'[!] collect {len(self.data)} samples and {counter} key-values')
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        ids = self.data[i]
        return torch.LongTensor(ids)

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }
