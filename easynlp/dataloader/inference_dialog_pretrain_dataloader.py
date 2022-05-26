from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class InferenceDRBERTDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.pad_token_id
        self.sep = self.vocab.sep_token_id
        self.cls = self.vocab.cls_token_id
        root_path = args['data_root_path']

        self.file_lists = [f'{root_path}/train_{i}.txt' for i in range(8)]
        random.shuffle(self.file_lists)
        self.current_file_index = 0
        self.current_file_handler = None
        self.cache = []
        self.buffer_size = args['buffer_size']

        # reset the random seed for each worker
        new_seed = args['seed'] + args['local_rank']
        random.seed(new_seed)
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)
                
    def __len__(self):
        # inference only a partial of the dataset
        return 2000

    def __getitem__(self, i):
        if len(self.cache) == 0:
            if self.current_file_handler is None:
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        line = self.cache.pop()
        line = json.loads(line)['data'][-1]
        items = self.vocab.encode(line, add_special_tokens=False)
        rids = [self.cls] + items[:self.args['res_max_len']] + [self.sep]
        return torch.LongTensor(rids), line

    def save(self):
        pass
        
    def collate(self, batch):
        rids = pad_sequence([i for i, j in batch], batch_first=True, padding_value=self.pad)
        text = [j for i, j in batch]
        rids_mask = generate_mask(rids)
        rids, rids_mask = to_cuda(rids, rids_mask)
        return {'ids': rids, 'mask': rids_mask, 'text': text}
