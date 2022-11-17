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
        # self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_{args["global_rank"]}.rar'
        # rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/searched_results_debug.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load train data by RandomAccessReader Object over: {self.reader.size}')
        else:
            if self.args['mode'] == 'train':
                self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_{args["global_rank"]}.txt')
                # self.reader = RandomAccessReader(f'{self.data_root_path}/searched_results_debug.txt')
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
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            # self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/backup_v4_data/searched_results_{i}.txt' for i in range(file_num)]
        # self.file_lists = [f'{self.data_root_path}/backup_v2_data/searched_results_debug.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            # new_seed = args['seed'] + args['local_rank']
            new_seed = args['seed'] + args['global_rank']
            random.seed(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker {self.args["local_rank"]}:')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

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
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())
            base_index = item['index']

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0 and item_o == self.base_data[docid[0]][docid[1]:docid[1]+length_s]:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            if len(ids) > 0:
                ids_total.append(torch.LongTensor(ids))
                vl.append(len(ids))
            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                if self.args['lang'] == 'en':
                    # bert-base-cased UNK replacement
                    phrase = phrase.replace('<unk>', '[UNK]')
                    phrase = phrase.replace('@,@', ',')
                    phrase = phrase.replace('@.@', '.')
                    phrase = phrase.replace('@-@', '-')

                    pre_phrase = pre_phrase.replace('<unk>', '[UNK]')
                    pre_phrase = pre_phrase.replace('@,@', ',')
                    pre_phrase = pre_phrase.replace('@.@', '.')
                    pre_phrase = pre_phrase.replace('@-@', '-')

                    post_phrase = post_phrase.replace('<unk>', '[UNK]')
                    post_phrase = post_phrase.replace('@,@', ',')
                    post_phrase = post_phrase.replace('@.@', '.')
                    post_phrase = post_phrase.replace('@-@', '-')
                phrase_ids = self.bert_vocab.encode(phrase, add_special_tokens=False)
                pre_phrase_ids = self.bert_vocab.encode(pre_phrase, add_special_tokens=False)
                post_phrase_ids = self.bert_vocab.encode(post_phrase, add_special_tokens=False)
                try:
                    self._truncate_triplet(pre_phrase_ids, phrase_ids, post_phrase_ids, self.args['doc_max_length'] - 2)
                except:
                    continue
                if base_index == docid:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.cls_token_id]
                else:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.sep_token_id]
                doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append((doc_s_pos, doc_e_pos))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }


class CopyGenerationCompactV2Dataset(Dataset):

    '''faster'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(args['phrase_tokenizer'])
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'

        self.size = 500
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(16)]
        # self.file_lists = [f'{self.data_root_path}/searched_results_debug.txt']
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
        ipdb.set_trace()
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
        token_pos_total = []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            token_pos = []
            for item_, docid in item['results'][self.last_delta:]:
                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, len(item_), len(items), docid[0], docid[1]))
                        # replace the last token with the end of the phrase
                        token_pos[-1] += len(items) 
                    else:
                        token_pos.extend([len(ids) + i for i in range(len(items))])
                else:
                    token_pos.extend([len(ids) + i for i in range(len(items))])
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
            token_pos.pop()
            token_pos_total.append(token_pos)

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
            print(f'[!] doc size: {len(ids_total)}', end='\r')

        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, token_pos_total

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, token_pos_total = batch[0]
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
            'token_pos_total': token_pos_total,
        }

class CopyGenerationOnlyGenWikiText103Dataset(Dataset):

    '''wikitext103 or en-wiki test set'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        if self.args['mode'] == 'train':
            self.size = iter_count(path)
            self.file_name = path
            print(path)
            self.file = open(path, 'r')
            self.max_len = args['max_len']
            self.cache = []
            self.buffer_size = 40960
            print(f'[!] size: {self.size}')
        else:
            self.data = []
            with open(path) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        tokens = self.vocab.encode(line, add_special_tokens=False)
                        for i in range(0, len(tokens), 512):
                            sequence = tokens[i:i+512]
                            if len(sequence) >= 128:
                                length = int(0.5 * len(sequence))
                                # label = [self.vocab.eos_token_id] * length + sequence[length:]
                                label = [self.vocab.pad_token_id] * length + sequence[length:]
                                assert len(label) == len(sequence)
                                self.data.append((sequence, label))
                self.size = len(self.data)
                print(f'[!] size: {self.size}')

    def __len__(self):
        return self.size

    def load_one_chunk(self):
        self.cache = load_lines_chunk(self.file, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.file = open(self.file_name, 'r')
            self.cache = load_lines_chunk(self.file, self.buffer_size)

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            if len(self.cache) == 0:
                self.load_one_chunk()
            text = self.cache.pop(0)
            # replace the <unk> to [UNK]
            # text = text.replace('<unk>', '[UNK]')
            text = text.replace('<unk>', '<|endoftext|>')
            ids = self.vocab.encode(text, add_special_tokens=False)
            if len(ids) > self.max_len:
                sample_arange = range(0, len(ids) - self.max_len)
                begin = random.choice(sample_arange)
                ids = ids[begin:begin+self.max_len]
            return ids
        else:
            tokens, label = self.data[i]
            return torch.LongTensor(tokens), torch.LongTensor(label)

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in batch]
            # ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            ids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id)
            ids, ids_mask = to_cuda(ids, ids_mask)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
            }
        else:
            assert len(batch) == 1
            ids, label = batch[0]
            ids = ids.unsqueeze(0).cuda()
            label = label.unsqueeze(0).cuda()
            return {
                'ids': ids,
                'ids_mask': torch.ones_like(ids),
                'label': label
            }



class CopyGenerationOnlyGenWikiText103TestDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        path = path.replace('test.txt', 'test_prefix.txt')
        # path = path.replace('test.txt', 'test_overfit_prefix.txt')

        self.data = []
        with open(path) as f:
            for line in f.readlines():
                item = json.loads(line)
                self.data.append(item)
            self.size = len(self.data)
            print(f'[!] size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        bundle = self.data[i]
        prefix = bundle['prefix']
        ground_truth = bundle['ground_truth']
        prefix = self.vocab.encode(prefix, add_special_tokens=False)[-(512-128):]
        return torch.LongTensor(prefix), bundle['ground_truth']

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, gt = batch[0]
        ids = ids.unsqueeze(0).cuda()
        return {
            'ids': ids,
            'gt': gt
        }

class CopyGenerationHoldoutCompactDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        # self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'

        self.path = f'{self.data_root_path}/test_overfit.txt'
        self.size = iter_count(self.path)

        self.current_file_index = 0
        self.current_file_handler = open(self.path, 'r')
        self.cache = []
        self.buffer_size = 1000
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            # self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.path, 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        if len(self.cache) == 0:
            self.load_one_chunk()
        item = json.loads(self.cache[0].strip())['results']
        text = [i[0] if idx == 0 else ' ' + i[0] for idx, i in enumerate(item)]
        text = ''.join(text)
        tokens = self.vocab.encode(text, add_special_tokens=False)
        self.cache.pop(0)
        return torch.LongTensor(tokens[:512])

    def save(self):
        pass
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.vocab.eos_token_id)
        ids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id)
        return {
            'ids': ids,
            'ids_mask': ids_mask
        }


class GPT2DomainAdaptionDataset(Dataset):

    '''小数据上进行fine-tune，数据格式和copyphrase的数据格式一样，一行一个document'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        if self.args['lang']:
            self.pad = self.vocab.eos_token_id
        else:
            self.pad = self.vocab.pad_token_id

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load data by RandomAccessReader Object over: {self.reader.size}')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] load data by RandomAccessReader Object over: {self.reader.size}')
        self.size = self.reader.size
        self.reader.init_file_handler()
        self.max_len = args['max_len']

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        item = self.reader.get_line(i)
        ids =  self.vocab.encode(item, add_special_tokens=False)
        if len(ids) > self.max_len:
            sample_arange = range(0, len(ids) - self.max_len)
            begin = random.choice(sample_arange)
            ids = ids[begin:begin+self.max_len]
        return ids

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [torch.LongTensor(i) for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }

class CopyGenerationCompactHNDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        # self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']], use_fast=True)
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
            file_num = 32
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(8)]
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        print(f'[!] load training datset: {self.size}')

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        for i in range(file_num):
            with open(f'{self.data_root_path}/searched_results_{i}_base.txt') as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    id_label = line['index']
                    chunk = line['results']
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over: {len(self.base_data)}')

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
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def clean_text(self, text, phrase=True):
        if self.args['lang'] == 'en':
            if phrase:
                text = text.replace('<unk>', '[UNK]')
            else:
                text = text.replace('<unk>', '<|endoftext|>')
            text = text.replace('@,@', ',')
            text = text.replace('@.@', '.')
            text = text.replace('@-@', '-')
        return text

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                # if self.args['lang'] == 'en':
                #     # replace the <unk> with <|endoftext|>
                #     item_ = item_.replace('<unk>', '<|endoftext|>')
                #     item_ = item_.replace('@,@', ',')
                #     item_ = item_.replace('@.@', '.')
                #     item_ = item_.replace('@-@', '-')
                item_ = self.clean_text(item_, phrase=False)
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:

                if docid not in self.base_data:
                    continue
                new_segments = self.update_segments(self.base_data[docid], item_o)
                if not new_segments:
                    # the item_o don't exist
                    continue

                item_o = self.clean_text(item_o)
                new_segments = [self.clean_text(seg) for seg in new_segments]
                doc_label_ = [True if seg == item_o else False for seg in new_segments]
                first_true_label = doc_label_.index(True)
                doc_ = self.bert_vocab.batch_encode_plus(
                    new_segments, 
                    add_special_tokens=False, 
                    return_attention_mask=False, 
                    return_token_type_ids=False
                )['input_ids']

                # doc_, doc_label_ = [], []
                # for seg in new_segments:
                #     if item_o == seg:
                #         doc_label_.append(True)
                #         if self.args['lang'] == 'en':
                #             item_o = item_o.replace('<unk>', '[UNK]')
                #             item_o = item_o.replace('@,@', ',')
                #             item_o = item_o.replace('@.@', '.')
                #             item_o = item_o.replace('@-@', '-')
                #         seg_ids = self.bert_vocab.encode(item_o, add_special_tokens=False)
                #         doc_.append(seg_ids)
                #     else:
                #         doc_label_.append(False)
                #         if self.args['lang'] == 'en':
                #             seg = seg.replace('<unk>', '[UNK]')
                #             seg = seg.replace('@,@', ',')
                #             seg = seg.replace('@.@', '.')
                #             seg = seg.replace('@-@', '-')
                #         seg_ids = self.bert_vocab.encode(seg, add_special_tokens=False)
                #         doc_.append(seg_ids)

                    
                doc_ids_, phrase_position = self.phrase_collect(first_true_label, doc_, doc_label_, self.args['doc_max_length'])
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append(phrase_position)
                
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def update_segments(self, segments, string):
        '''string may exist in multiple segments'''
        left_pos, right_pos = [], []
        sep = ' ' if self.args['lang'] == 'en' else ''
        sentence = sep.join(segments)

        # cannot find the phrases in the document, just ignore it
        try:
            index = sentence.index(string)
            string_b_pos = index
            string_e_pos = index + len(string)
        except:
            return []
        counter = 0
        for idx, seg in enumerate(segments):
            if idx > 0 and self.args['lang'] == 'en':
                counter += 1
            left_pos.append(counter)
            counter += len(seg)
            right_pos.append(counter)

        seg_labels = []
        new_segments = []
        for l, r, s in zip(left_pos, right_pos, segments):
            if r <= string_b_pos or l >= string_e_pos:
                seg_labels.append(0)
                new_segments.append(s)
            else:
                if len(seg_labels) > 0 and seg_labels[-1] == 1:
                    new_segments[-1] += sep + s
                else:
                    new_segments.append(s)
                    seg_labels.append(1)
        assert len(seg_labels) == len(new_segments)

        ss = []
        for seg, l in zip(new_segments, seg_labels):
            if l == 1:
                index = seg.index(string)
                pre = seg[:index].strip()
                post = seg[index+len(string):].strip()
                if pre:
                    ss.append(pre)
                ss.append(string)
                if post:
                    ss.append(post)
            else:
                ss.append(seg)
        return ss

    def phrase_collect(self, first_true_label, phrases, phrases_label, max_length):
        '''return the position index of the phrases in the documents; and the token ids of the documents which is truncated by the max_length'''
        ids = [phrases[first_true_label]]
        item_o = phrases[first_true_label]
        hn_label = [True]
        left_boundary = first_true_label - 1
        right_boundary = first_true_label + 1
        last_index = 0
        max_length -= 2
        while sum([len(i) for i in ids]) < max_length:
            if left_boundary < 0 and right_boundary < len(phrases):
                # left illegal and right legal
                last_index = len(ids)
                ids.append(phrases[right_boundary])
                hn_label.append(phrases_label[right_boundary])
                right_boundary += 1
            elif left_boundary >= 0 and right_boundary >= len(phrases):
                # left legal and right illegal
                last_index = 0
                ids[0:0] = [phrases[left_boundary]]
                hn_label[0:0] = [phrases_label[left_boundary]]
                left_boundary -= 1
            elif left_boundary < 0 or right_boundary >= len(phrases):
                # both are illegal, just break the loop
                break
            else:
                # left and right are legal, select the phrases that have the smaller length
                if len(phrases[left_boundary]) < len(phrases[right_boundary]):
                    last_index = 0
                    ids[0:0] = [phrases[left_boundary]]
                    hn_label[0:0] = [phrases_label[left_boundary]]
                    left_boundary -= 1
                else:
                    last_index = len(ids)
                    ids.append(phrases[right_boundary])
                    hn_label.append(phrases_label[right_boundary])
                    right_boundary += 1
        cur_length = sum([len(i) for i in ids])
        if cur_length > max_length:
            delta = cur_length - max_length
            if last_index == 0:
                ids[last_index] = ids[last_index][delta:]
            else:
                ids[last_index] = ids[last_index][:-delta]

        phrase_positions = []
        doc_ids = [self.bert_vocab.cls_token_id]
        flag = False
        for seg, l in zip(ids, hn_label):
            begin_index = len(doc_ids)
            doc_ids.extend(seg)
            end_index = len(doc_ids) - 1
            if l:
                if flag is False:
                    flag = True
                    phrase_positions.append((begin_index, end_index, l))
            else:
                if self.args['min_phrase_length'] <= len(seg) <= self.args['max_phrase_length']:
                    phrase_positions.append((begin_index, end_index, l))
        doc_ids.append(self.bert_vocab.sep_token_id)
        return doc_ids, phrase_positions

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask= to_cuda(ids, dids, ids_mask, dids_mask)
        return {
            'ids': ids, 
            'dids': dids, 
            'doc_index': dindex,
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }



class CopyGenerationCompactHNFastDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
            file_num = 32
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(8)]
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        print(f'[!] load training datset: {self.size}')

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        for i in range(file_num):
            with open(f'{self.data_root_path}/searched_results_{i}_base.txt') as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    id_label = line['index']
                    chunk = line['results']
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over: {len(self.base_data)}')

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
        random.shuffle(self.cache)

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
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_, doc_label_ = [], []

                if docid not in self.base_data:
                    continue
                new_segments = self.update_segments(self.base_data[docid], item_o)
                if not new_segments:
                    # the item_o don't exist
                    continue

                for seg in new_segments:
                    if item_o == seg:
                        doc_label_.append(True)
                        if self.args['lang'] == 'en':
                            item_o = item_o.replace('<unk>', '[UNK]')
                            item_o = item_o.replace('@,@', ',')
                            item_o = item_o.replace('@.@', '.')
                            item_o = item_o.replace('@-@', '-')
                        seg_ids = self.bert_vocab.encode(item_o, add_special_tokens=False)
                        doc_.append(seg_ids)
                    else:
                        doc_label_.append(False)
                        if self.args['lang'] == 'en':
                            seg = seg.replace('<unk>', '[UNK]')
                            seg = seg.replace('@,@', ',')
                            seg = seg.replace('@.@', '.')
                            seg = seg.replace('@-@', '-')

                        seg_ids = self.bert_vocab.encode(seg, add_special_tokens=False)
                        doc_.append(seg_ids)

                try:
                    first_true_label = doc_label_.index(True)
                except:
                    ipdb.set_trace()
                    
                doc_ids_, phrase_position = self.phrase_collect(first_true_label, doc_, doc_label_, self.args['doc_max_length'])
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append(phrase_position)
                
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def update_segments(self, segments, string, doc_max_length):
        '''string may exist in multiple segments'''
        left_pos, right_pos = [], []
        sep = ' ' if self.args['lang'] == 'en' else ''
        sentence = sep.join(segments)

        # cannot find the phrases in the document, just ignore it
        try:
            index = sentence.index(string)
            string_b_pos = index
            string_e_pos = index + len(string)
        except:
            return []
        counter = 0
        for idx, seg in enumerate(segments):
            if idx > 0 and self.args['lang'] == 'en':
                counter += 1
            left_pos.append(counter)
            counter += len(seg)
            right_pos.append(counter)

        seg_labels = []
        new_segments = []
        for l, r, s in zip(left_pos, right_pos, segments):
            if r <= string_b_pos or l >= string_e_pos:
                seg_labels.append(0)
                new_segments.append(s)
            else:
                if len(seg_labels) > 0 and seg_labels[-1] == 1:
                    new_segments[-1] += sep + s
                else:
                    new_segments.append(s)
                    seg_labels.append(1)
        assert len(seg_labels) == len(new_segments)
        first_true_label = seg_labels.index(1)

        ss, doc_label_ = [], []
        for seg, l in zip(new_segments, seg_labels):
            if l == 1:
                index = seg.index(string)
                pre = seg[:index].strip()
                post = seg[index+len(string):].strip()
                if pre:
                    ss.append(pre)
                    doc_label_.append(False)
                ss.append(string)
                doc_label_.append(True)
                if post:
                    ss.append(post)
                    doc_label_.append(False)
            else:
                ss.append(seg)
                doc_label_.append(False)

        ss = self.bert_vocab.batch_encode_plus(ss, add_special_tokens=False)['input_ids']
        return ss

    def phrase_collect(self, first_true_label, phrases, phrases_label, max_length):
        '''return the position index of the phrases in the documents; and the token ids of the documents which is truncated by the max_length'''
        ids = [phrases[first_true_label]]
        item_o = phrases[first_true_label]
        hn_label = [True]
        left_boundary = first_true_label - 1
        right_boundary = first_true_label + 1
        last_index = 0
        max_length -= 2
        while sum([len(i) for i in ids]) < max_length:
            if left_boundary < 0 and right_boundary < len(phrases):
                # left illegal and right legal
                last_index = len(ids)
                ids.append(phrases[right_boundary])
                hn_label.append(phrases_label[right_boundary])
                right_boundary += 1
            elif left_boundary >= 0 and right_boundary >= len(phrases):
                # left legal and right illegal
                last_index = 0
                ids[0:0] = [phrases[left_boundary]]
                hn_label[0:0] = [phrases_label[left_boundary]]
                left_boundary -= 1
            elif left_boundary < 0 or right_boundary >= len(phrases):
                # both are illegal, just break the loop
                break
            else:
                # left and right are legal, select the phrases that have the smaller length
                if len(phrases[left_boundary]) < len(phrases[right_boundary]):
                    last_index = 0
                    ids[0:0] = [phrases[left_boundary]]
                    hn_label[0:0] = [phrases_label[left_boundary]]
                    left_boundary -= 1
                else:
                    last_index = len(ids)
                    ids.append(phrases[right_boundary])
                    hn_label.append(phrases_label[right_boundary])
                    right_boundary += 1
        cur_length = sum([len(i) for i in ids])
        if cur_length > max_length:
            delta = cur_length - max_length
            if last_index == 0:
                ids[last_index] = ids[last_index][delta:]
            else:
                ids[last_index] = ids[last_index][:-delta]

        phrase_positions = []
        doc_ids = [self.bert_vocab.cls_token_id]
        flag = False
        for seg, l in zip(ids, hn_label):
            begin_index = len(doc_ids)
            doc_ids.extend(seg)
            end_index = len(doc_ids) - 1
            if l:
                if flag is False:
                    flag = True
                    phrase_positions.append((begin_index, end_index, l))
            else:
                phrase_positions.append((begin_index, end_index, l))
        doc_ids.append(self.bert_vocab.sep_token_id)
        return doc_ids, phrase_positions

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask= to_cuda(ids, dids, ids_mask, dids_mask)
        return {
            'ids': ids, 
            'dids': dids, 
            'doc_index': dindex,
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }

class CopyGenerationCompactOnlyPhraseDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

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
        random.shuffle(self.cache)

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        # while len(doc_ids) < self.args["max_doc_size"]:
        while len(doc_ids) < self.args["max_doc_size"] and len(ids_total) < self.args['max_sample_num']:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0: 
                        item_doc = item_o.replace('<unk>', '[UNK]')
                        item_doc = item_doc.replace('@,@', ',')
                        item_doc = item_doc.replace('@.@', '.')
                        item_doc = item_doc.replace('@-@', '-')
                        phrase_ids = self.bert_vocab.encode(item_doc)
                        docs.append((counter - 1, len(items), phrase_ids))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_i, phrase_ids in docs:
                doc_ids.append(torch.LongTensor(phrase_ids))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, pos_ids, pos_ids_end, vl = batch[0]
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask = to_cuda(ids, dids, ids_mask, dids_mask)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }



class CopyGenerationCompactOnlyPhraseHNDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_hn_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"] and len(ids_total) < self.args['max_sample_num']:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                phrase = item_o.replace('<unk>', '[UNK]')
                phrase = phrase.replace('@,@', ',')
                phrase = phrase.replace('@.@', '.')
                phrase = phrase.replace('@-@', '-')
                phrase_ids = self.bert_vocab.encode(phrase)
                doc_ = self.base_data[docid]
                select_random = random.random()

                over = False
                if select_random <= 0.5:
                    # move the right
                    select_range = range(
                        pos_in_doc+length_s+4, 
                        min(len(doc_), pos_in_doc+length_s+16)
                    )
                    if len(select_range) > 0:
                        over = True
                        select_right_index = random.choice(select_range)
                        phrase_hn = doc_[pos_in_doc:select_right_index]
                else:
                    # move the left
                    select_range = range(
                        max(0, pos_in_doc-16), 
                        max(0, pos_in_doc)
                    )
                    if len(select_range) > 0:
                        over = True
                        select_left_index = random.choice(select_range)
                        phrase_hn = doc_[select_left_index:pos_in_doc+length_s]
                if over is False:
                    # random select the phrase in the document
                    # shift the position
                    random_left = random.choice(range(len(doc_) - 2))
                    random_range = random.choice(range(min(32, len(doc_) - random_left)))
                    phrase_hn = doc_[random_left:random_left+random_range]

                phrase_hn = phrase_hn.replace('<unk>', '[UNK]')
                phrase_hn = phrase_hn.replace('@,@', ',')
                phrase_hn = phrase_hn.replace('@.@', '.')
                phrase_hn = phrase_hn.replace('@-@', '-')
                phrase_hn_ids = self.bert_vocab.encode(phrase_hn)

                doc_ids.append(torch.LongTensor(phrase_ids))
                doc_hn_ids.append(torch.LongTensor(phrase_hn_ids))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        doc_ids.extend(doc_hn_ids)
        return ids_total, doc_ids, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, pos_ids, pos_ids_end, vl = batch[0]
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask = to_cuda(ids, dids, ids_mask, dids_mask)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }


class CopyGenerationCompactOnlyPhraseHNV2Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        # self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']], use_fast=True)
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
            file_num = 32
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(8)]
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        print(f'[!] load training datset: {self.size}')

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        for i in range(file_num):
            with open(f'{self.data_root_path}/searched_results_{i}_base.txt') as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    id_label = line['index']
                    chunk = line['results']
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over: {len(self.base_data)}')

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
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def clean_text(self, text, phrase=True):
        if self.args['lang'] == 'en':
            if phrase:
                text = text.replace('<unk>', '[UNK]')
            else:
                text = text.replace('<unk>', '<|endoftext|>')
            text = text.replace('@,@', ',')
            text = text.replace('@.@', '.')
            text = text.replace('@-@', '-')
        return text

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_hn_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                item_ = self.clean_text(item_, phrase=False)
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, len(items), docid[0], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_i, docid, item_o in docs:
                if docid not in self.base_data:
                    continue
                new_segments = self.update_segments(self.base_data[docid], item_o)
                if not new_segments:
                    # the item_o don't exist
                    continue

                item_o = self.clean_text(item_o)
                phrase_ids = self.bert_vocab.encode(item_o)
                doc_ids.append(torch.LongTensor(phrase_ids))
                new_segments = [self.clean_text(seg) for seg in new_segments]
                if self.args['lang'] == 'zh':
                    hn_segments = [seg for seg in new_segments if seg != item_o and self.args['min_phrase_length'] <= len(seg) <= self.args['max_phrase_length']]
                else:
                    hn_segments = [seg for seg in new_segments if seg != item_o and self.args['min_phrase_length'] <= len(seg.split()) <= self.args['max_phrase_length']]
                sample_num = min(len(hn_segments), self.args['gray_cand_num'])
                hn_segments = random.sample(hn_segments, sample_num)
                if len(hn_segments):
                    phrase_hn_ids = self.bert_vocab.batch_encode_plus(hn_segments)['input_ids']
                    phrase_hn_ids = [torch.LongTensor(i) for i in phrase_hn_ids]
                    doc_hn_ids.extend(phrase_hn_ids)
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        doc_ids.extend(doc_hn_ids)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def update_segments(self, segments, string):
        '''string may exist in multiple segments'''
        left_pos, right_pos = [], []
        sep = ' ' if self.args['lang'] == 'en' else ''
        sentence = sep.join(segments)

        # cannot find the phrases in the document, just ignore it
        try:
            index = sentence.index(string)
            string_b_pos = index
            string_e_pos = index + len(string)
        except:
            return []
        counter = 0
        for idx, seg in enumerate(segments):
            if idx > 0 and self.args['lang'] == 'en':
                counter += 1
            left_pos.append(counter)
            counter += len(seg)
            right_pos.append(counter)

        seg_labels = []
        new_segments = []
        for l, r, s in zip(left_pos, right_pos, segments):
            if r <= string_b_pos or l >= string_e_pos:
                seg_labels.append(0)
                new_segments.append(s)
            else:
                if len(seg_labels) > 0 and seg_labels[-1] == 1:
                    new_segments[-1] += sep + s
                else:
                    new_segments.append(s)
                    seg_labels.append(1)
        assert len(seg_labels) == len(new_segments)

        ss = []
        for seg, l in zip(new_segments, seg_labels):
            if l == 1:
                index = seg.index(string)
                pre = seg[:index].strip()
                post = seg[index+len(string):].strip()
                if pre:
                    ss.append(pre)
                ss.append(string)
                if post:
                    ss.append(post)
            else:
                ss.append(seg)
        return ss

    def phrase_collect(self, first_true_label, phrases, phrases_label, max_length):
        '''return the position index of the phrases in the documents; and the token ids of the documents which is truncated by the max_length'''
        ids = [phrases[first_true_label]]
        item_o = phrases[first_true_label]
        hn_label = [True]
        left_boundary = first_true_label - 1
        right_boundary = first_true_label + 1
        last_index = 0
        max_length -= 2
        while sum([len(i) for i in ids]) < max_length:
            if left_boundary < 0 and right_boundary < len(phrases):
                # left illegal and right legal
                last_index = len(ids)
                ids.append(phrases[right_boundary])
                hn_label.append(phrases_label[right_boundary])
                right_boundary += 1
            elif left_boundary >= 0 and right_boundary >= len(phrases):
                # left legal and right illegal
                last_index = 0
                ids[0:0] = [phrases[left_boundary]]
                hn_label[0:0] = [phrases_label[left_boundary]]
                left_boundary -= 1
            elif left_boundary < 0 or right_boundary >= len(phrases):
                # both are illegal, just break the loop
                break
            else:
                # left and right are legal, select the phrases that have the smaller length
                if len(phrases[left_boundary]) < len(phrases[right_boundary]):
                    last_index = 0
                    ids[0:0] = [phrases[left_boundary]]
                    hn_label[0:0] = [phrases_label[left_boundary]]
                    left_boundary -= 1
                else:
                    last_index = len(ids)
                    ids.append(phrases[right_boundary])
                    hn_label.append(phrases_label[right_boundary])
                    right_boundary += 1
        cur_length = sum([len(i) for i in ids])
        if cur_length > max_length:
            delta = cur_length - max_length
            if last_index == 0:
                ids[last_index] = ids[last_index][delta:]
            else:
                ids[last_index] = ids[last_index][:-delta]

        phrase_positions = []
        doc_ids = [self.bert_vocab.cls_token_id]
        flag = False
        for seg, l in zip(ids, hn_label):
            begin_index = len(doc_ids)
            doc_ids.extend(seg)
            end_index = len(doc_ids) - 1
            if l:
                if flag is False:
                    flag = True
                    phrase_positions.append((begin_index, end_index, l))
            else:
                if self.args['min_phrase_length'] <= len(seg) <= self.args['max_phrase_length']:
                    phrase_positions.append((begin_index, end_index, l))
        doc_ids.append(self.bert_vocab.sep_token_id)
        return doc_ids, phrase_positions

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask= to_cuda(ids, dids, ids_mask, dids_mask)
        return {
            'ids': ids, 
            'dids': dids, 
            'doc_index': dindex,
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }



class CopyGenerationCompactNoContextDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

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
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                if self.args['lang'] == 'en':
                    # bert-base-cased UNK replacement
                    phrase = phrase.replace('<unk>', '<|endoftext|>')
                    phrase = phrase.replace('@,@', ',')
                    phrase = phrase.replace('@.@', '.')
                    phrase = phrase.replace('@-@', '-')

                    pre_phrase = pre_phrase.replace('<unk>', '<|endoftext|>')
                    pre_phrase = pre_phrase.replace('@,@', ',')
                    pre_phrase = pre_phrase.replace('@.@', '.')
                    pre_phrase = pre_phrase.replace('@-@', '-')

                    post_phrase = post_phrase.replace('<unk>', '<|endoftext|>')
                    post_phrase = post_phrase.replace('@,@', ',')
                    post_phrase = post_phrase.replace('@.@', '.')
                    post_phrase = post_phrase.replace('@-@', '-')
                phrase_ids = self.bert_vocab.encode(phrase, add_special_tokens=False)
                pre_phrase_ids = self.bert_vocab.encode(pre_phrase, add_special_tokens=False)
                post_phrase_ids = self.bert_vocab.encode(post_phrase, add_special_tokens=False)
                try:
                    self._truncate_triplet(pre_phrase_ids, phrase_ids, post_phrase_ids, self.args['doc_max_length'] - 2)
                except:
                    continue
                doc_ids_ = pre_phrase_ids + phrase_ids + post_phrase_ids
                # reverse it
                doc_ids_ = list(reversed(doc_ids_))
                doc_e_pos, doc_s_pos = len(post_phrase_ids) - 1, len(post_phrase_ids) + len(phrase_ids) - 1
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append((doc_s_pos, doc_e_pos))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.eos_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.eos_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }



class CopyGenerationCompactV3Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

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
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                if self.args['lang'] == 'en':
                    # bert-base-cased UNK replacement
                    phrase = phrase.replace('<unk>', '[UNK]')
                    phrase = phrase.replace('@,@', ',')
                    phrase = phrase.replace('@.@', '.')
                    phrase = phrase.replace('@-@', '-')

                    pre_phrase = pre_phrase.replace('<unk>', '[UNK]')
                    pre_phrase = pre_phrase.replace('@,@', ',')
                    pre_phrase = pre_phrase.replace('@.@', '.')
                    pre_phrase = pre_phrase.replace('@-@', '-')

                    post_phrase = post_phrase.replace('<unk>', '[UNK]')
                    post_phrase = post_phrase.replace('@,@', ',')
                    post_phrase = post_phrase.replace('@.@', '.')
                    post_phrase = post_phrase.replace('@-@', '-')
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
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }



class CopyGenerationCompactV3Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)
        # self.file_lists = [f'{self.data_root_path}/test_overfit.txt' for i in range(8)]
        # self.size = iter_count(self.file_lists[0])

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def clean_text(self, text, phrase=True):
        if self.args['lang'] == 'en':
            if phrase:
                text = text.replace('<unk>', '[UNK]')
            else:
                text = text.replace('<unk>', '<|endoftext|>')
            text = text.replace('@,@', ',')
            text = text.replace('@.@', '.')
            text = text.replace('@-@', '-')
        return text

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())

            # collect documents
            docs, ids, phrase_counter, counter, delta_ = [], [], len(self.vocab), 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                length_s = len(item_)
                item_ = self.clean_text(item_, phrase=False)
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_
                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    docs.append((length_s, docid[0], docid[1]))
                    ids.append(phrase_counter)
                    phrase_counter += 1
                    counter += 1
                else:
                    ids.extend(items)
                    counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            for length_s, docid, pos_in_doc in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                # bert-base-cased UNK replacement
                phrsae = self.clean_text(phrase)
                pre_phrsae = self.clean_text(pre_phrase)
                post_phrsae = self.clean_text(post_phrase)
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
        return ids_total, doc_ids, doc_index, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
        }



class CopyGenerationCompactWithPrefixDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_tokenizer'][args['lang']])
        if self.args['dataset'] == 'wikitext103':
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
            file_num = 8
        else:
            self.data_root_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data'
            file_num = 8
        self.file_lists = [f'{self.data_root_path}/searched_results_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['local_rank']
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker (self.args["local_rank"]):')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                # chunk = ' [SEP] '.join(line[:-1])
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
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

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
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())
            sample_index = item['index']
            original_doc = self.base_data[sample_index]

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                # only for engish
                length_s = len(item_)
                item_o = item_

                if self.args['lang'] == 'en':
                    # replace the <unk> with <|endoftext|>
                    item_ = item_.replace('<unk>', '<|endoftext|>')
                    item_ = item_.replace('@,@', ',')
                    item_ = item_.replace('@.@', '.')
                    item_ = item_.replace('@-@', '-')
                if self.args['lang'] == 'en' and counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    if random.random() <= self.args['replace_ratio']:
                        # replace with the original document
                        if counter > 0:
                            try:
                                docs.append((counter - 1, length_s, len(items), sample_index, original_doc.index(item_o), item_o))
                            except:
                                # cannot find the phrase in the original document, just ignore it
                                pass
                    else:
                        docid = docid[0]
                        if counter > 0:
                            docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            ids_total.append(torch.LongTensor(ids))
            vl.append(len(ids))

            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                if self.args['lang'] == 'en':
                    # bert-base-cased UNK replacement
                    phrase = phrase.replace('<unk>', '[UNK]')
                    phrase = phrase.replace('@,@', ',')
                    phrase = phrase.replace('@.@', '.')
                    phrase = phrase.replace('@-@', '-')

                    pre_phrase = pre_phrase.replace('<unk>', '[UNK]')
                    pre_phrase = pre_phrase.replace('@,@', ',')
                    pre_phrase = pre_phrase.replace('@.@', '.')
                    pre_phrase = pre_phrase.replace('@-@', '-')

                    post_phrase = post_phrase.replace('<unk>', '[UNK]')
                    post_phrase = post_phrase.replace('@,@', ',')
                    post_phrase = post_phrase.replace('@.@', '.')
                    post_phrase = post_phrase.replace('@-@', '-')
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
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }


