from header import *
from itertools import islice
from .utils import *
from .util_func import *
from .randomaccess import *
from .check import *


class PhrasesInferenceDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/searched_results_inference.txt'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        item = json.loads(self.reader.get_line(i))
        data = item['results']
        ids, pos, texts = [], [], []
        for text, docs in data:
            ids_ = self.vocab.encode(text, add_special_tokens=False)
            if 2 + len(ids) + len(ids_) > self.args['max_len']:
                break
            if len(docs) == 0 or \
                is_all_chinese(text) is False or \
                length_check(text, self.args['inf_phrase_min_len'], self.args['inf_phrase_max_len']) is False:
                ids.extend(ids_)
            else:
                b = len(ids)
                ids.extend(ids_)
                e = len(ids) - 1
                pos.append((b, e))
                texts.append(text)
        if len(pos) == 0:
            return None, None, None
        ids = torch.LongTensor([self.cls] + ids + [self.sep]) 
        return ids, pos, texts

    def save(self):
        pass
        
    def collate(self, batch):
        ids, pos, text = [], [], []
        for a, b, c in batch:
            if a is None:
                continue
            ids.append(a)
            pos.append(b)
            text.append(c)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'pos': pos,
            'text': text
        }
