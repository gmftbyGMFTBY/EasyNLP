from header import *
from .utils import *
from .util_func import *
from .randomaccess import *


class DialogSimCTGDataset(Dataset):

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
        return 208779677

    def __getitem__(self, i):
        if len(self.cache) == 0:
            if self.current_file_handler is None:
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            random.shuffle(self.cache)
        line = self.cache.pop()
        line = json.loads(line)['data']
        items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
        context_ids = []
        response_ids = items[-1]
        for s in items[:-1]:
            context_ids.extend(s + [self.sep])
        context_ids.pop()
        truncate_pair(context_ids, response_ids, self.args['max_len'])
        ids = [self.cls] + context_ids + [self.sep] + response_ids + [self.sep]
        return torch.LongTensor(ids)

    def save(self):
        pass
        
    def collate(self, batch):
        ids = pad_sequence([i for i in batch if i is not None], batch_first=True, padding_value=self.pad)
        ids, ods = ids[:, :-1], ids[:, 1:]
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ods, ids_mask = to_cuda(ids, ods, ids_mask)
        return {'ids': ids, 'ods': ods, 'ids_mask': ids_mask}


class DialogEVADataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
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
        return 208779677

    def __getitem__(self, i):

        if len(self.cache) == 0:
            if self.current_file_handler is None:
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            random.shuffle(self.cache)
        line = self.cache.pop()
        line = json.loads(line)['data']
        items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
        context, response = items[:-1], items[-1]
        ids = []
        for s in context:
            ids.extend(s + [self.sep])
        ids.pop()
        ids = [self.cls] + ids[-self.args['max_len']:] + [self.sep]
        response = [self.cls] + response[:self.args['res_max_len']] + [self.sep]
        return torch.LongTensor(ids), torch.LongTensor(response)

    def save(self):
        pass
        
    def collate(self, batch):
        input_ids = [i for i, j in batch]
        output_ids = [j for i, j in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.pad)
        input_ids_mask = generate_mask(input_ids)
        output_ids_mask = generate_mask(output_ids)
        output_ids, labels = output_ids[:, :-1], output_ids[:, 1:]
        output_ids_mask = output_ids_mask[:, :-1]
        input_ids, input_ids_mask, output_ids, output_ids_mask, labels = to_cuda(input_ids, input_ids_mask, output_ids, output_ids_mask, labels)
        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'input_ids_mask': input_ids_mask,
            'output_ids_mask': output_ids_mask,
            'labels': labels
        }

class DialogPLATOV1Dataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOU]', '[BOU]'])
        self.pad = self.vocab.pad_token_id
        self.sep = self.vocab.sep_token_id
        self.bou, self.eou = self.vocab.convert_tokens_to_ids(['[BOU]', '[EOU]'])
        self.mask = self.vocab.mask_token_id

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
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            random.shuffle(self.cache)

        line = self.cache.pop()
        line = json.loads(line)['data']
        items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
        context_ids, response_ids = [], items[-1]
        role_ids, turn_ids, role, turn_index = [], [], 0, 0
        for s in items[:-1]:
            context_ids.extend(s + [self.eou])
            role_ids.extend([role] * (len(s) + 1))
            turn_ids.extend([turn_index] * (len(s) + 1))
            turn_index += 1
            role = 0 if role == 1 else 1

        # random negative sample for response selection training
        if len(self.cache) == 0:
            if self.current_file_handler is None:
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            if len(self.cache) == 0:
                # curretn file runs over, move to next file
                self.current_file_index = 0 if self.current_file_index + 1 > 7 else self.current_file_index + 1
                self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
                print(f'[!] open new file {self.file_lists[self.current_file_index]}')
                self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
            random.shuffle(self.cache)
        random_item = json.loads(random.choice(self.cache))['data']
        random_sentence = random.choice(random_item)
        neg_response_ids = self.vocab.encode(random_sentence, add_special_tokens=False)

        # positive 
        res_role_ids, res_turn_ids = [role] * len(response_ids), [turn_index] * len(response_ids)
        context_ids_ = deepcopy(context_ids)
        role_ids_ = deepcopy(role_ids)
        turn_ids_ = deepcopy(turn_ids)
        self.truncate_pair(
            context_ids_, 
            response_ids, 
            role_ids_, 
            turn_ids_, 
            res_role_ids,
            res_turn_ids,
            self.args['max_len']
        )
        pos_role_ids = role_ids_ + [res_role_ids[0]] + res_role_ids + [res_role_ids[0]]
        pos_turn_ids = turn_ids_ + [res_turn_ids[0]] + res_turn_ids + [res_turn_ids[0]]
        pos_ids = [self.mask] + context_ids_ + [self.bou] + response_ids + [self.eou]
        pos_response_length = len(response_ids) + 2
        pos_length = len(pos_ids)

        # negative
        res_role_ids, res_turn_ids = [role] * len(neg_response_ids), [turn_index] * len(neg_response_ids)
        context_ids_ = deepcopy(context_ids)
        role_ids_ = deepcopy(role_ids)
        turn_ids_ = deepcopy(turn_ids)
        self.truncate_pair(
            context_ids_, 
            neg_response_ids, 
            role_ids_, 
            turn_ids_, 
            res_role_ids,
            res_turn_ids,
            self.args['max_len']
        )
        neg_role_ids = role_ids_ + [res_role_ids[0]] + res_role_ids + [res_role_ids[0]]
        neg_turn_ids = turn_ids_ + [res_turn_ids[0]] + res_turn_ids + [res_turn_ids[0]]
        neg_ids = [self.mask] + context_ids_ + [self.bou] + neg_response_ids + [self.eou]
        neg_response_length = len(neg_response_ids) + 2
        neg_length = len(neg_ids)
        return (
            torch.LongTensor(pos_ids), 
            torch.LongTensor(pos_role_ids), 
            torch.LongTensor(pos_turn_ids),
            torch.LongTensor(neg_ids), 
            torch.LongTensor(neg_role_ids), 
            torch.LongTensor(neg_turn_ids),
            pos_response_length,
            pos_length,
            neg_response_length,
            neg_length,
        )

    def truncate_pair(
        self, 
        context_ids, 
        response_ids, 
        role_ids,
        turn_ids,
        res_role_ids,
        res_turn_ids,
        max_length,
    ):
        max_length -= 3    # [MASK]/z, [BOU], [EOU]
        while True:
            l = len(context_ids) + len(response_ids)
            if l <= max_length:
                break
            if len(context_ids) > 2 * len(response_ids):
                context_ids.pop(0)
                role_ids.pop(0)
                turn_ids.pop(0)
            else:
                response_ids.pop()
                res_role_ids.pop()
                res_turn_ids.pop()

    def save(self):
        pass
        
    def collate(self, batch):
        ids, role_ids, turn_ids = [], [], []
        neg_ids, neg_role_ids, neg_turn_ids = [], [], []
        pos_response_length, pos_length = [], []
        neg_response_length, neg_length = [], []
        for a, b, c, d, e, f, g, h, i, j in batch:
            ids.append(a)
            role_ids.append(b)
            turn_ids.append(c)
            neg_ids.append(d)
            neg_role_ids.append(e)
            neg_turn_ids.append(f)
            pos_response_length.append(g)
            pos_length.append(h)
            neg_response_length.append(i)
            neg_length.append(j)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        role_ids = pad_sequence(role_ids, batch_first=True, padding_value=self.pad)
        turn_ids = pad_sequence(turn_ids, batch_first=True, padding_value=self.pad)
        neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
        neg_role_ids = pad_sequence(neg_role_ids, batch_first=True, padding_value=self.pad)
        neg_turn_ids = pad_sequence(neg_turn_ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        neg_ids_mask = generate_mask(neg_ids, pad_token_idx=self.pad)
        ids, role_ids, turn_ids, ids_mask = to_cuda(ids, role_ids, turn_ids, ids_mask)
        neg_ids, neg_role_ids, neg_turn_ids, neg_ids_mask = to_cuda(neg_ids, neg_role_ids, neg_turn_ids, neg_ids_mask)
        return {
            'ids': ids, 
            'role_ids': role_ids, 
            'turn_ids': turn_ids, 
            'ids_mask': ids_mask,
            'neg_ids': neg_ids, 
            'neg_role_ids': neg_role_ids, 
            'neg_turn_ids': neg_turn_ids, 
            'neg_ids_mask': neg_ids_mask,
            'pos_response_length': pos_response_length,
            'pos_length': pos_length,
            'neg_response_length': neg_response_length,
            'neg_length': neg_length,
        }
