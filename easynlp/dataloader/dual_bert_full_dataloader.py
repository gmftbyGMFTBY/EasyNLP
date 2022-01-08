from header import *
from .utils import *
from .randomaccess import *
from .util_func import *


class BERTDualFullWithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.data = []
        if self.args['mode'] == 'train':
            self.path_name = path
            responses = []
            with open(path) as f:
                pbar = tqdm(f.readlines())
                for line in pbar:
                    line = [i.strip() for i in json.loads(line.strip())['nr'] if i.strip()]
                    responses.extend(line)
                    if len(responses) > 1000000:
                        break
                    pbar.set_description(f'[!] already collect {len(responses)} utterances for candidates')
            self.responses = list(set(responses))
            print(f'[!] load {len(self.responses)} utterances')
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.rar'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccessReader Object over')
            else:
                self.reader = RandomAccessReader(self.path_name)
                self.reader.init()
                torch.save(self.reader, rar_path)
                print(f'[!] save the random access reader file into {rar_path}')
            self.reader.init_file_handler()
            self.size = self.reader.size
            print(f'[!] dataset size: {self.size}')
        else:
            data, responses = read_json_data(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                items = self.vocab.batch_encode_plus(context+[response]+candidates, add_special_tokens=False)['input_ids']
                cids, rids, hrids = items[:len(context)], items[len(context)], items[-len(candidates):]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = [self.cls] + ids[-(self.args['max_len']-2):] + [self.sep]
                rids = [[self.cls] + r[:(self.args['res_max_len']-2)] + [self.sep] for r in [rids] + hrids]
                self.data.append({
                    'label': [1] + [0] * 9,
                    'ids': ids,
                    'rids': rids
                })
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i)
            line = json.loads(line.strip())
            q, r, nr = line['q'], line['r'], line['nr']
            nr = [i for i in nr if i.strip()]
            if len(nr) < self.args['gray_cand_num']:
                nr += random.sample(self.responses, self.args['gray_cand_num']-len(nr))
            else:
                nr = random.sample(nr, self.args['gray_cand_num'])

            items = self.vocab.batch_encode_plus(q + [r] + nr, add_special_tokens=False)['input_ids']
            cids, rids, hrids = items[:len(q)], items[len(q)], items[-len(nr):]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = [self.cls] + ids[-(self.args['max_len'])-2:] + [self.sep]
            rids = [rids] + hrids
            rids_ = []
            for r in rids:
                rids_.append([self.cls] + r[:(self.args['res_max_len']-2)] + [self.sep])
            return ids, rids_
        else:
            bundle = self.data[i]
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i[0]) for i in batch]
            rids = []
            for _, b in batch:
                rids.extend([torch.LongTensor(i) for i in b])
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
            assert len(batch) == 1
            ids, rids, label = batch[0]
            label = torch.LongTensor(label)
            ids, rids, label = to_cuda(ids, rids, label)
            return {
                'ids': ids,
                'rids': rids,
                'label': label,
            }
