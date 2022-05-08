from header import *
from .utils import *
from .util_func import *
from .randomaccess import *
from itertools import accumulate


class CopyGenerationDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(args['phrase_tokenizer'])
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'

        if self.args['mode'] == 'train':
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
        else:
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_0.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over')
        else:
            if self.args['mode'] == 'train':
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
            else:
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_31.txt')
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' [SEP] '.join(line[:-1])
                id_label = line[-1]
                base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

    def __len__(self):
        return self.size

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def __getitem__(self, i):
        item = json.loads(self.reader.get_line(i))

        if self.args['mode'] == 'test':
            index, string = 0, ''
            length = sum([len(i) for i, _ in item['results']])
            prefix_length = int(length * self.args['prefix_length_rate'])
            for idx, (i, _) in enumerate(item['results']):
                if len(string) + len(i) > prefix_length:
                    index = idx
                    break
                string += i
            prefix = ''.join([i for i, _ in item['results'][:index]])
            ground_truth = ''.join([i for i, _ in item['results'][index:]])
            return prefix, ground_truth

        # random sample the start index
        lengths = [len(i) for i, _ in item['results']]
        lengths = list(accumulate(lengths))
        lengths = lengths[-1] - np.array(lengths)
        index_range = 0
        for idx, l in enumerate(lengths):
            if l < self.args['max_len']:
                break
            index_range = idx
        try:
            start_index = random.sample(range(0, index_range))
        except:
            start_index = 0
        
        docs, ids, counter = [], [], 0
        for item_, docid in item['results'][start_index:]:
            items = self.vocab.encode(item_, add_special_tokens=False)
            if len(ids) + len(items) > self.args['max_len']:
                break
            if docid:
                docid = docid[0]
                if counter > 0:
                    docs.append((counter - 1, len(item_), len(items), docid[0], docid[1]))
            ids.extend(items)
            counter += len(items)

        if len(docs) > 0 and counter - docs[-1][0] <= 2:
            # dangerous case
            docs.pop()

        if len(docs) > self.args['max_doc_num']:
            docs = random.sample(docs, self.args['max_doc_num'])

        # encode the documents
        doc_ids, doc_index, pos_index, pos_index_end = [], [], [], []
        for pos_in_ids, length_s, length_i, docid, pos_in_doc in docs:
            doc_ = self.base_data[docid]
            pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
            phrase = doc_[pos_in_doc:pos_in_doc+length_s]
            phrase_ids = self.bert_vocab.encode(phrase, add_special_tokens=False)
            pre_phrase_ids = self.bert_vocab.encode(pre_phrase, add_special_tokens=False)
            post_phrase_ids = self.bert_vocab.encode(post_phrase, add_special_tokens=False)
            try:
                self._truncate_triplet(pre_phrase_ids, phrase_ids, post_phrase_ids, self.args['doc_max_length'] - 2)
            except:
                continue

            doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.sep_token_id]
            doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), 1 + len(pre_phrase_ids) + len(phrase_ids)
            doc_ids.append(doc_ids_)
            doc_index.append((doc_s_pos, doc_e_pos))
            pos_index.append(pos_in_ids)
            pos_index_end.append(pos_in_ids + length_i)
        return ids, doc_ids, doc_index, pos_index, pos_index_end

    def save(self):
        pass
        
    def collate(self, batch):

        if self.args['mode'] == 'test':
            assert len(batch) == 1
            batch = batch[0]
            return {
                'prefix': batch[0],
                'ground_truth': batch[1]
            }

        ids, dids, dindex_s, dindex_e, pos_ids, pos_ids_end = [], [], [], [], [], []
        for a, b, c, d, e in batch:
            ids.append(torch.LongTensor(a))
            dids.extend([torch.LongTensor(i) for i in b])
            dindex_s.extend([i for i, _ in c])
            dindex_e.extend([i for _, i in c])
            pos_ids.append(d)
            pos_ids_end.append(e)
        dindex_s = torch.LongTensor(dindex_s)
        dindex_e = torch.LongTensor(dindex_e)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
        ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }


class CopyGenerationOnlyGenDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over')
        else:
            if self.args['mode'] == 'train':
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
            else:
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_31.txt')
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        self.max_len = args['max_len']

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        item = json.loads(self.reader.get_line(i))
        ids = []
        for item, _ in item['results']:
            items = self.vocab.encode(item, add_special_tokens=False)
            ids.extend(items)
        if len(ids) > self.max_len:
            sample_arange = range(0, len(ids) - self.max_len)
            begin = random.choice(sample_arange)
            ids = ids[begin:begin+self.max_len]
        return ids

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [torch.LongTensor(i) for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }
