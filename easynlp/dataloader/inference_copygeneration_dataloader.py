from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class InferenceCopyGenerationDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(args['phrase_tokenizer'])
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/inference_data_collect/inference_collection_results.pt'
        results = torch.load(self.data_root_path)
        self.results = self.filter_results(results)
        base_data = {}
        with open('/apdcephfs/share_916081/johntianlan/copygeneration_data/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' [SEP] '.join(line[:-1])
                id_label = line[-1]
                base_data[id_label] = chunk
        self.base_data = base_data
        self.index = list(base_data.keys())
        print(f'[!] load base data over')

    def filter_results(self, results, min_len=5):
        new = {}
        counter = 0
        for key, value in tqdm(results.items()):
            values = [(pos, l) for pos, l in value if l >= min_len]
            values = random.sample(list(set(values)), 5)
            new[key] = values
            counter += len(values)
        print(f'[!] find {counter} phrases')
        return new

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        index = self.index[i]
        doc = self.base_data[index]
        phrases = self.results[index]
        phrases = sorted(phrases, key=lambda x:x[0])
        
        segments = []
        labels = []
        begin = 0
        for pos, l in phrases:
            segment = doc[begin:pos]
            if segment:
                segments.append(segment)
                labels.append(0)
            
            segment = doc[pos:pos+l]
            segments.append(segment)
            labels.append(1)
            begin = pos + l

        if begin < len(doc):
            segments.append(doc[begin:])
            labels.append(0)

        assert ''.join(segments) == doc
        assert len(labels) == len(segments)

        ids, pos, text = [], [], []
        for seg, label in zip(segments, labels):
            if len(ids) + len(item) + 2 > self.args['max_len']:
                break
            item = self.vocab.encode(seg, add_special_tokens=False)
            begin_pos = len(ids)
            ids.extend(item)
            end_pos = len(ids)
            if label == 1:
                pos.append((begin_pos, end_pos - 1))
                text.append(seg)
        ids = torch.LongTensor([self.cls] + ids + [self.sep])
        return ids, pos, text

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [i for i, _, _ in batch]
        pos = [i for _, i, _ in batch]
        text = [i for _, _, i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'phrase_pos': pos,
            'phrase_text': text,
        }
