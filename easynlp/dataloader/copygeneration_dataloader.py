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
            # rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_debug.rar'
        else:
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_0.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
        else:
            if self.args['mode'] == 'train':
                # self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_debug.txt')
            else:
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_31.txt')
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
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

        # if len(docs) > self.args['max_doc_num']:
        #     docs = random.sample(docs, self.args['max_doc_num'])
        #     docs = sorted(docs, key=lambda x:x[0])

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
            doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
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
        # rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_debug.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
        else:
            if self.args['mode'] == 'train':
                # self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_debug.txt')
            else:
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_31.txt')
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
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


class CopyGenerationGADataset(Dataset):

    '''support the gradient accumulation'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(args['phrase_tokenizer'])
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'

        if self.args['mode'] == 'train':
            # rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_debug.rar'
        else:
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_0.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
        else:
            if self.args['mode'] == 'train':
                # self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_debug.txt')
            else:
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_31.txt')
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
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

        # if len(docs) > self.args['max_doc_num']:
        #     docs = random.sample(docs, self.args['max_doc_num'])
        #     docs = sorted(docs, key=lambda x:x[0])

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
            doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
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
        return batch

        ids, dids, dindex_s, dindex_e, pos_ids, pos_ids_end = [], [], [], [], [], []
        for a, b, c, d, e in batch:
            ids.append(torch.LongTensor(a))
            dids.extend([torch.LongTensor(i) for i in b])
            dindex_s.extend([i for i, _ in c])
            dindex_e.extend([i for _, i in c])
            pos_ids.append(d)
            pos_ids_end.append(e)
        # dindex_s = torch.LongTensor(dindex_s)
        # dindex_e = torch.LongTensor(dindex_e)
        # ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        # dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
        # ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        # ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
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



class CopyGenerationCompactDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(args['phrase_tokenizer'])
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'

        self.size = 500000
        # self.file_lists = [f'{self.args["data_root_path"]}/searched_results_{i}.txt' for i in range(32)]
        self.file_lists = [f'{self.data_root_path}/searched_results_debug.txt']
        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        new_seed = args['seed'] + args['local_rank']
        random.seed(new_seed)
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)
        random.shuffle(self.file_lists)

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

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)

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
        # read till the max bert dids are achieved
        ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, len(item_), len(items), docid[0], docid[1]))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta = delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
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
                doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append((doc_s_pos, doc_e_pos))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
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
