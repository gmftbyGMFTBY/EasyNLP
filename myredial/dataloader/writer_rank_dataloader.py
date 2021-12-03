from header import *
from .randomaccess import *
from .utils import *
from .util_func import *
from .augmentation import *


class WriterRankDataset(Dataset):

    '''for gpt2 inference'''

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
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.rar'
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
        while True:
            line = self.reader.get_line(i)
            sentences = json.loads(line.strip())['q']
            sentences = [s.strip() for s in sentences if s.strip()]    # ignore the empty sentence
            if self.filter(sentences):
                break
            i = random.choice(range(self.size))

        sentences = [''.join(sentence.split()) for sentence in sentences]
        if random.random() < 0.5:
            # half of the possibility: use the context-response pair
            res_idx = random.randint(1, len(sentences)-1)
            context, response = sentences[:res_idx], sentences[res_idx]
        else:
            # half of the possibility: uncomplete context-response pair
            res_idx = random.randint(0, len(sentences)-1)
            length = len(sentences[res_idx])
            idx = random.randint(int(0.25 * length), int(0.5 * length))
            context = sentences[:res_idx] + [sentences[res_idx][:idx]]
            response = sentences[res_idx][idx:]
        # convert the text into the token ids
        cids, rids = [], []
        for s in context:
            cids.extend(self.vocab.encode(s, add_special_tokens=False))
        rids.extend(self.vocab.encode(response, add_special_tokens=False))
        # max_length includes the ctx_length and res_length
        if self.args['model'] in ['bert-ft-writer', 'writer-electra']:
            truncate_pair(cids, rids, self.args['gpt2_max_len'])
        else:
            cids = cids[-self.args['gpt2_max_len']:]
        return cids, rids

    def collate(self, batch):
        gpt2_cids, cids, rids = [], [], []
        max_length = -1
        for a, b in batch:
            cids.append(a)
            gpt2_cids.extend([a] * self.args['inference_time'])
            rids.append(b)
            max_length = max(max_length, len(a))
        # pad to the max_length
        gpt2_cids = torch.stack(
            [torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in gpt2_cids]
        )
        gpt2_cids_mask = torch.stack(
            [torch.LongTensor([0]*(max_length-len(i)) + [1]*len(i)) for i in gpt2_cids]
        )
        gpt2_pos_ids = (gpt2_cids_mask.long().cumsum(-1) - 1).masked_fill(gpt2_cids_mask == 0, 0)
        gpt2_cids, gpt2_cids_mask, gpt2_pos_ids = to_cuda(gpt2_cids, gpt2_cids_mask, gpt2_pos_ids)

        # replace some negative samples
        random_index = random.sample(range(self.size), self.args['easy_cand_pool_size'])
        easy_rids = []
        for idx in random_index:
            while True:
                line = self.reader.get_line(idx)
                sentences = json.loads(line.strip())['q']
                sentences = [s.strip() for s in sentences if s.strip()]    # ignore the empty sentence
                if self.filter(sentences):
                    break
                idx = random.choice(range(self.size))
            easy_rids.append(self.vocab.encode(random.choice(sentences), add_special_tokens=False))
        return {
           'gpt2_cids': gpt2_cids, 
           'gpt2_cids_mask': gpt2_cids_mask, 
           'gpt2_pos_ids': gpt2_pos_ids,
           'cids': cids,
           'rids': rids,
           'erids': easy_rids,
        }

    def filter(self, sentences):
        '''filter out the bad documents'''
        line = ''.join(sentences)
        # url is bad
        if line.count('http') > 0:
            return False
        if len(sentences) <= 1:
            return False
        return True
