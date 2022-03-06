from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualSessionDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.gray_cand_num = args['gray_cand_num']

        self.data = []
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}_gray_session.rar'
        path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}_gray.txt'
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
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i)
            item = json.loads(line.strip())
            ctx, res, cands = item['q'], item['r'], item['hp']
            cand = random.choice(cands)

            utterances = ctx + [res, cand]
            tokens = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids = tokens[:len(ctx)]
            rids = tokens[-2]
            crids = tokens[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            truncate_pair(ids, crids, self.args['max_len'])
            ids = [self.cls] + ids + [self.eos] + crids + [self.sep]

            rids = rids[:(self.args['res_max_len']-2)]
            rids = [self.cls] + rids + [self.sep]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            return ids, rids
        else:
            line = self.reader.get_line(i)
            item = json.loads(line.strip())
            ctx, responses, cands = item['q'], item['r'], item['hp']
            cands = list(set(cands) - set([r for _, r in responses]))
            res, label = [], []
            for l, r in responses:
                res.append(r)
                label.append(l)

            utterances = ctx + res + cands
            tokens = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids = tokens[:len(ctx)]
            rids = tokens[len(ctx):len(ctx)+len(res)]
            crids = tokens[-len(cands):]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()

            tokens = []
            rids = [torch.LongTensor([self.cls] + r[:(self.args['res_max_len']-2)] + [self.sep]) for r in rids]
            for crids_ in crids:
                ids_ = deepcopy(ids)
                truncate_pair(ids_, crids_, self.args['max_len'])
                ids_ = [self.cls] + ids_ + [self.eos] + crids_ + [self.sep]
                tokens.append(torch.LongTensor(ids_))
            return tokens, rids, label

    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = [i[1] for i in batch]
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
            ids, rids, label = batch[0]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            ids_mask = generate_mask(ids)
            label = torch.LongTensor(label)
            ids, ids_mask, rids, rids_mask, label = to_cuda(ids, ids_mask, rids, rids_mask, label)
            return {
                'ids': ids, 
                'ids_mask': ids_mask,
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
            }
