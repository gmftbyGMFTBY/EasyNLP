from header import *
from .utils import *
from .util_func import *
from .randomaccess import *
from config import *
from model import *


class PostTrainBigDataset(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])
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
        return 208779677

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
            random.shuffle(self.cache)
        line = self.cache.pop()
        session = json.loads(line)['data']
        session = self.vocab.batch_encode_plus(session, add_special_tokens=False)['input_ids']
        tokens = []
        for utterance in session[:-1]:
            tokens.extend(utterance + [self.eos])
        tokens.pop()

        ratio = random.random()
        if ratio > 0.75:
            # ground-truth
            response = session[-1]
            label = 2
        elif ratio > 0.5:
            # within session
            response = random.choice(session[:-1])
            label = 1
        else:
            if len(self.cache) <= 1000:
                if self.current_file_handler is None:
                    self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                self.cache += load_lines_chunk(self.current_file_handler, self.buffer_size)
                random.shuffle(self.cache)
            response = random.choice(self.cache)
            response = random.choice(json.loads(response)['data'])
            response = self.vocab.encode(response, add_special_tokens=False)
            label = 0

        try:
            response_ = deepcopy(response)
            truncate_pair(tokens, response_, self.args['max_len'])
            ids = [self.cls] + tokens + [self.sep] + response_ + [self.sep]
            tids = [0] * (len(tokens) + 2) + [1] * (len(response_) + 1)
            mask_labels = mask_sentence(
                ids,
                self.args['min_mask_num'], 
                self.args['max_mask_num'], 
                self.args['masked_lm_prob'], 
                special_tokens=self.special_tokens, 
                mask=self.mask, 
                vocab_size=len(self.vocab),
            )
        except:
            return None, None, None, None
        return ids, tids, mask_labels, label

    def save(self):
        pass
        
    def collate(self, batch):
        ids, tids, mask_labels, labels = [], [], [], []
        for ids_, tids_, mask_labels_, labels_ in batch:
            if ids_ is None:
                continue
            ids.append(ids_)
            tids.append(tids_)
            mask_labels.append(mask_labels_)
            labels.append(labels_)
        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        labels = torch.LongTensor(labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, mask_labels, attn_mask, labels = to_cuda(ids, tids, mask_labels, attn_mask, labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
            'label': labels,
        }


class PostTrainMonoBigDatasetV2(Dataset):

    '''Dynamic Mask: no mask token will be set as the -1 label
    For chinese corpus, the train.txt and test.txt must have been tokenzied by the white space'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask, self.eos])
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
        return 208779677

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
            random.shuffle(self.cache)
        line = self.cache.pop()
        sentences = json.loads(line)['data']
        try:
            max_l = -1
            for s in sentences:
                if len(s) > max_l:
                    max_l = len(s)
                    sentence = s
            ids = [self.cls] + self.vocab.encode(sentence, add_special_tokens=False)[:self.args['max_len']-2] + [self.sep]
            mask_label = mask_sentence(
                ids,
                self.args['min_mask_num'], 
                self.args['max_mask_num'], 
                self.args['masked_lm_prob'], 
                special_tokens=self.special_tokens, 
                mask=self.mask, 
                vocab_size=len(self.vocab),
            )
            return {
                'ids': ids,
                'mask_labels': mask_label
            }
        except Exception as error:
            print(error)
            return None

    def save(self):
        pass
        
    def collate(self, batch):
        ids, mask_labels = [], []
        for item in batch:
            if item is None:
                continue
            ids.append(item['ids'])
            mask_labels.append(item['mask_labels'])
        ids = [torch.LongTensor(i) for i in ids]
        mask_labels = [torch.LongTensor(i) for i in mask_labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, mask_labels, attn_mask = to_cuda(ids, mask_labels, attn_mask)
        return {
            'ids': ids, 
            'mask_labels': mask_labels, 
            'attn_mask': attn_mask, 
        }


