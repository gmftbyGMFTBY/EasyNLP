from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class SimCSEDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        rar_path = path.replace('train.txt', 'train.rar')
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')

        # if self.args['dataset'] in ['chinese_wiki']:
        #     data = []
        #     with open(path) as f:
        #         for line in tqdm(f.readlines()):
        #             data.append(json.loads(line.strip())['q'])
        #     data = list(set(data))
        # else:
        #     # data = read_text_data_utterances(path, lang=self.args['lang'])
        #     # data = list(chain(*[u for label, u in data if label == 1]))
        #     # data = list(set(data))
        #     data = []
        #     with open(path) as f:
        #         for line in tqdm(f.readlines()):
        #             data.append(line.strip())
        # print(f'[!] collect {len(data)} samples for simcse')
        # self.data = []
        # for idx in tqdm(range(0, len(data), 32)):
        #     utterances = data[idx:idx+32]
        #     item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
        #     ids = [[self.cls] + i[:self.args["res_max_len"]-2] + [self.sep] for i in item]
        #     self.data.extend(ids)
                
    def __len__(self):
        return self.reader.size

    def __getitem__(self, i):
        item = self.reader.get_line(i).strip()
        tokens = self.vocab.encode(item, add_special_tokens=False)
        ids = [self.cls] + tokens[:self.args['res_max_len']-2] + [self.sep]
        ids = torch.LongTensor(ids)
        return ids

    def save(self):
        pass
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }


class BERTSimCSEInferenceDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            path = f'{args["root_dir"]}/data/{self.args["dataset"]}/base_data.txt'
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        items = line.split('\t')
        text_id = items[-1]
        line = ''.join(items[:-1])
        rids = self.vocab.encode(line, add_special_tokens=False)
        rids = [self.cls] + rids[:self.args['max_len']-2] + [self.sep]
        rid = torch.LongTensor(rids)
        return rid, text_id

    def save(self):
        pass
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text_id = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text_id': rid_text_id
        }

        
class BERTSimCSEInferenceContextDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            path = f'{args["root_dir"]}/data/{self.args["dataset"]}/base_data_nore.txt'
            self.reader = RandomAccessReader(path)
            self.reader.init()
            torch.save(self.reader, rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        items = line.split('\t')
        text_id = items[-1]
        line = ''.join(items[:-1])
        rids = self.vocab.encode(line, add_special_tokens=False)
        rids = [self.cls] + rids[:self.args['max_len']-2] + [self.sep]
        rid = torch.LongTensor(rids)
        return rid, text_id

    def save(self):
        pass
        
    def collate(self, batch):
        ids, text, index = [], [], []
        for i, j, k in batch:
            ids.append(i)
            index.append(k)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'index': index,
        }

        
class SimCSEUnlikelyhoodDataset(Dataset):

    '''training dataset'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_simcse_unlikelyhood_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_unlikelyhood(path, length=16)
        print(f'[!] collect {len(data)} samples for simcse training')
        self.data = []
        for idx in tqdm(range(0, len(data), 256)):
            utterances = data[idx:idx+256]
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = [[self.cls] + i[:self.args["res_max_len"]-2] + [self.sep] for i in item]
            self.data.extend(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = torch.LongTensor(self.data[i])
        return ids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }

        
class BERTSimCSEUnlikelyhoodInferenceContextDataset(Dataset):

    '''Only for test dataset, generate the negative sample for calculating the ppl'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_unlikelyhood_simcse_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        path = f'{os.path.split(path)[0]}/test.txt'
        dataset = read_text_data_unlikelyhood_test(path)
        self.data = []
        counter = 0
        for utterance in tqdm(dataset):
            utterance = ''.join(utterance.strip().split())
            sentences_ = [i.strip() for i in re.split('(。|，|！|？|，)', utterance) if i.strip()]
            sentences = []
            for i in sentences_:
                if i in ['。', '，', '！', '？'] and len(sentences) > 0:
                    sentences[-1] += i
                else:
                    sentences.append(i)
            item = self.vocab.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']
            item = [[self.cls] + i[:self.args['max_len']] + [self.sep] for i in item]

            for idx in range(1, len(item)):
                ids = item[idx]
                utterance = ''.join(sentences[:idx])
                response = sentences[idx]
                for fix in ['。', '，', '！', '？']:
                    response = response.strip(fix)
                if len(response) < args['min_test_len'] or len(utterance) < args['min_test_context_len']:
                    continue
                self.data.append({
                    'ids': ids, 
                    # text is the context
                    'context': utterance,
                    'pos_res': response,
                    'index': counter
                })
            counter += 1
        print(f'[!] collect {len(self.data)} test samples')
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        return ids, bundle['context'], bundle['pos_res'], bundle['index']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, context, response, index = [], [], [], []
        for i, j, k, y in batch:
            ids.append(i)
            context.append(j)
            response.append(k)
            index.append(y)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'context': context,
            'response': response,
            'index': index,
        }

        
class BERTSimCSEUnlikelyhoodInferenceDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_unlikelyhood_simcse_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        dataset = read_text_data_unlikelyhood(path)
        # test set faiss index doesn't need so much
        dataset = random.sample(list(set(dataset)), 1000000)
        self.data = []
        for utterance in tqdm(dataset):
            rids = length_limit_res(self.vocab.encode(utterance), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': utterance,
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text,
        }

class InferenceWZSimCSEDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if self.args['mode'] in ['train', 'inference']:
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccesReader Object over')
            else:
                self.reader = RandomAccessReader(path)
                self.reader.init()
                torch.save(self.reader, rar_path)
            self.reader.init_file_handler()
            self.size = self.reader.size
        else:
            dataset = []
            with open(path) as f:
                for line in f.readlines():
                    items = line.strip().split('\t')
                    assert len(items) == 3
                    s1, s2, l = items
                    dataset.append((s1, s2, int(l)))
            self.data = dataset
            self.data = self.data[:10000]
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # random sample a data point
        if self.args['mode'] in ['train', 'inference']:
            line = self.reader.get_line(i).strip()
            return line
        else:
            s1, s2, l = self.data[i]
            return s1, s2, l

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] in ['train', 'inference']:
            output = self.vocab(batch, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            ids = output['input_ids']
            ids_mask = output['attention_mask']
            tids = output['token_type_ids']
            ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
            return {
                'ids': ids, 
                'tids': tids,
                'ids_mask': ids_mask, 
                'text': batch
            }
        else:
            s1, s2, l = [], [], []
            for a, b, c in batch:
                s1.append(a)
                s2.append(b)
                l.append(c)
            output = self.vocab(s1, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            s1_ids = output['input_ids']
            s1_ids_mask = output['attention_mask']
            s1_tids = output['token_type_ids']
            output = self.vocab(s2, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            s2_ids = output['input_ids']
            s2_ids_mask = output['attention_mask']
            s2_tids = output['token_type_ids']
            s1_ids, s1_tids, s1_ids_mask, s2_ids, s2_tids, s2_ids_mask = to_cuda(s1_ids, s1_tids, s1_ids_mask, s2_ids, s2_tids, s2_ids_mask)
            return {
                's1_ids': s1_ids,
                's1_tids': s1_tids,
                's1_ids_mask': s1_ids_mask,
                's2_ids': s2_ids,
                's2_tids': s2_tids,
                's2_ids_mask': s2_ids_mask,
                'label': l
            }
            
class SupervisedInferenceWZSimCSEDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if self.args['mode'] in ['train', 'inference']:
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}_sup.rar'
            path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}_sup.txt'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccesReader Object over')
            else:
                self.reader = RandomAccessReader(path)
                self.reader.init()
                torch.save(self.reader, rar_path)
            self.reader.init_file_handler()
            self.size = self.reader.size
        else:
            dataset = []
            with open(path) as f:
                for line in f.readlines():
                    items = line.strip().split('\t')
                    assert len(items) == 3
                    s1, s2, l = items
                    dataset.append((s1, s2, int(l)))
            self.data = dataset
            self.data = self.data[:10000]
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # random sample a data point
        if self.args['mode'] in ['train', 'inference']:
            line = self.reader.get_line(i).strip()
            items = line.strip().split('\t')
            assert len(items) == 3
            _, s1, s2 = items
            return s1, s2
        else:
            s1, s2, l = self.data[i]
            return s1, s2, l

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] in ['train', 'inference']:
            ids_1 = [i[0] for i in batch]
            ids_2 = [i[1] for i in batch]
            output = self.vocab(ids_1, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            ids = output['input_ids']
            ids_mask = output['attention_mask']
            tids = output['token_type_ids']
            output = self.vocab(ids_2, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            ids_2 = output['input_ids']
            ids_mask_2 = output['attention_mask']
            tids_2 = output['token_type_ids']
            ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
            ids_2, tids_2, ids_mask_2 = to_cuda(ids_2, tids_2, ids_mask_2)
            return {
                'ids': ids, 
                'tids': tids,
                'ids_mask': ids_mask, 
                'ids_2': ids_2, 
                'tids_2': tids_2,
                'ids_mask_2': ids_mask_2, 
                'text': batch
            }
        else:
            s1, s2, l = [], [], []
            for a, b, c in batch:
                s1.append(a)
                s2.append(b)
                l.append(c)
            output = self.vocab(s1, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            s1_ids = output['input_ids']
            s1_ids_mask = output['attention_mask']
            s1_tids = output['token_type_ids']
            output = self.vocab(s2, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
            s2_ids = output['input_ids']
            s2_ids_mask = output['attention_mask']
            s2_tids = output['token_type_ids']
            s1_ids, s1_tids, s1_ids_mask, s2_ids, s2_tids, s2_ids_mask = to_cuda(s1_ids, s1_tids, s1_ids_mask, s2_ids, s2_tids, s2_ids_mask)
            return {
                's1_ids': s1_ids,
                's1_tids': s1_tids,
                's1_ids_mask': s1_ids_mask,
                's2_ids': s2_ids,
                's2_tids': s2_tids,
                's2_ids_mask': s2_ids_mask,
                'label': l
            }

class BERTSimCSEInferenceBigDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.size = 4848348
        local_rank = self.args['global_rank']
        local_rank = '0' + str(local_rank) if local_rank < 10 else str(local_rank)
        self.file = open(f'{args["root_dir"]}/data/{args["dataset"]}/inference_part_{local_rank}')
        print(f'[!] load file {self.file}')
        self.buffer = 10000
        self.cache = []

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if len(self.cache) == 0:
            self.cache = load_lines_chunk(self.file, self.buffer)
            if len(self.cache) == 0:
                return None, None
        line = self.cache.pop(0).strip()
        items = line.split('\t')
        text_id = items[-1]
        line = ' '.join(items[:-1])
        rids = self.vocab.encode(line, add_special_tokens=False)
        rids = [self.cls] + rids[:self.args['max_len']-2] + [self.sep]
        rid = torch.LongTensor(rids)
        return rid, text_id

    def save(self):
        pass
        
    def collate(self, batch):
        rid = [i[0] for i in batch if i[0] is not None]
        if len(rid) == 0:
            return None
        rid_text_id = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text_id': rid_text_id
        }



class SimCSEDialogDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.data = []
        path = path.replace('train.txt', 'train_cross_domain.txt') 
        print(f'[!] load data from {path}')
        with open(path) as f:
            for line in tqdm(f.readlines()):
                items = line.strip().split('\t')
                context = items[1:-2]
                item = self.vocab.batch_encode_plus(context, add_special_tokens=False)['input_ids']
                # ids = []
                # for i in item:
                #     ids.extend(i + [self.sep])
                # ids.pop()
                # ids = [self.cls] + ids[-self.args['max_len']+2:] + [self.sep]
                ids = [self.cls] + item[-1][:self.args['res_max_len']-2] + [self.sep]
                self.data.append(ids)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.LongTensor(self.data[i])

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



class BERTSimCSEInferenceDialogContextDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.data = []
        path = path.replace('train.txt', 'train_cross_domain.txt') 
        print(f'[!] load data from {path}')
        with open(path) as f:
            for line in tqdm(f.readlines()):
                items = line.strip().split('\t')
                context = items[1:-2]
                item = self.vocab.batch_encode_plus(context, add_special_tokens=False)['input_ids']
                ids = []
                for i in item:
                    ids.extend(i + [self.sep])
                ids.pop()
                ids = [self.cls] + ids[-self.args['max_len']+2:] + [self.sep]
                ids = [self.cls] + item[-1][:self.args['res_max_len']-2] + [self.sep]

                self.data.append({
                    'ids': ids,
                    'index': items[0],
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.LongTensor(self.data[i]['ids']), self.data[i]['index']

    def save(self):
        pass
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        text = [i[1] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'index': text
        }



