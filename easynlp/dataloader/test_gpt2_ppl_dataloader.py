from header import *
from .utils import *
from .util_func import *


class TestGPT2PPLDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        # for english model
        if self.args['lang'] == 'en':
            # self.pad = self.vocab.eos_token_id
            self.pad = self.vocab.pad_token_id
        else:
            self.pad = self.vocab.pad_token_id
        self.data = []
        data = read_text_data_line_by_line(path)
        self.data = []
        for line in tqdm(data):
            ids = self.vocab.encode(line, add_special_tokens=False)[:512]
            if len(ids) < 64:
                continue
            self.data.append({
                'ids': ids,
            })
        # self.data = self.data[:50]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = torch.LongTensor(self.data[i]['ids'])
        return ids

    def save(self):
        pass
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids,
            'ids_mask': ids_mask
        }
