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

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_rar.txt'
        path = f'{args["root_dir"]}/data/{args["dataset"]}/data.txt'
        self.reader = RandomAccessReader(path)
        self.reader.load_from_text(rar_path)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # try:
        line = json.loads(self.reader.get_line(i))['data']
        items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
        ids = []
        for s in items:
            ids.extend(s + [self.sep])
        ids.pop()
        ids = ids[-self.args['max_len']:]
        return torch.LongTensor(ids)
        # except:
        #     return None

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

        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_rar.txt'
        path = f'{args["root_dir"]}/data/{args["dataset"]}/data.txt'
        self.reader = RandomAccessReader(path)
        self.reader.load_from_text(rar_path, size=10000)
        self.size = self.reader.size
        self.reader.init_file_handler()
        print(f'[!] dataset size: {self.size}')
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # try:
        line = json.loads(self.reader.get_line(i))['data']
        items = self.vocab.batch_encode_plus(line, add_special_tokens=False)['input_ids']
        context, response = items[:-1], items[-1]
        ids = []
        for s in context:
            ids.extend(s + [self.sep])
        ids.pop()
        ids = [self.cls] + ids[-self.args['max_len']:] + [self.sep]
        response = [self.cls] + response[:self.args['res_max_len']] + [self.sep]
        return torch.LongTensor(ids), torch.LongTensor(response)
        # except:
        #     return None

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
