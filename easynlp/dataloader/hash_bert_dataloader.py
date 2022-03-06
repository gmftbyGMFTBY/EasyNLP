from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualHashFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.rar'
            path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'
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
        else:
            self.data = []
            data = read_text_data_utterances(path, lang=self.args['lang'])
            if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
                data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                rtext = []
                for label, utterances in batch:
                    ctext = ' '.join(utterances[:-1])
                    rtext.append(utterances[-1])
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                    'ctext': [ctext],
                    'rtext': rtext,
                })    
            self.size = len(self.data)
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = json.loads(self.reader.get_line(i).strip())
            q, r, hn = line['q'], line['r'], line['hp']
            if random.random() < 0.4:
                r = random.choice([r] + hn)
            items = self.vocab.batch_encode_plus(q + [r], add_special_tokens=False)['input_ids']
            cids, rids = items[:-1], items[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]
            rids = rids[:(self.args['res_max_len']-2)]
            ids = [self.cls] + ids + [self.sep]
            rids = [self.cls] + rids + [self.sep]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            return ids, rids
        else:
            bundle = self.data[i]
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text,
            }
