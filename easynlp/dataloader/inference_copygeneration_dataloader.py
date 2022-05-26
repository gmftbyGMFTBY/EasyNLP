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

class PhrasesInferenceV2Dataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.pad_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id

        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'

        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(dist.get_world_size())]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        self.file = f'{self.data_root_path}/searched_results_{self.args["global_rank"]}.txt'
        # self.file = f'{self.data_root_path}/inference.txt'
        size = iter_count(self.file)
        print(f'[!] file size for worker ({self.args["global_rank"]}): {self.file} ({size})')

        self.current_file_index = 0
        self.current_file_handler = open(self.file, 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1]
                base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

    def __len__(self):
        return self.size

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            return

    def __len__(self):
        return self.size

    def __getitem__(self, i):

        if len(self.cache) == 0:
            self.load_one_chunk()
            if len(self.cache) == 0:
                return None, None, None, None

        item = json.loads(self.cache.pop(0))
        data = item['results']
        source_index = item['index']
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
                b = len(ids) + 1    # [CLS] token
                ids.extend(ids_)
                e = len(ids)
                pos.append((b, e))
                texts.append(text)
        if len(pos) == 0:
            return None, None, None, None
        ids = torch.LongTensor([self.cls] + ids + [self.sep]) 
        return ids, pos, texts, [source_index] * len(pos)

    def save(self):
        pass
        
    def collate(self, batch):
        ids, pos, text, source_index = [], [], [], []
        for a, b, c, d in batch:
            if a is None:
                continue
            ids.append(a)
            pos.append(b)
            text.append(c)
            source_index.extend(d)
        if len(ids) == 0:
            return None
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'pos': pos,
            'text': text,
            'source': source_index 
        }
