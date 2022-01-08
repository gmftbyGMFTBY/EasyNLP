from header import *
from .utils import *
from .util_func import *
from .augmentation import *


class MutualTrainingDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_mutual_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_dpr_gray(f'{args["root_dir"]}/data/{args["dataset"]}/train_dpr_gray.txt')
            for item in tqdm(data):
                self.data.append({
                    'context': item['q'],
                    'response': item['r'],
                    'candidates': item['snr'],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': batch[0][1][:-1],
                    'responses': [b[1][-1] for b in batch]
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            candidates = random.sample(bundle['candidates'], self.args['gray_cand_num'])
            easy = [i['response'] for i in random.sample(self.data, self.args['gray_cand_num'])]
            return bundle['context'], bundle['response'], candidates, easy
        else:
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            context = [i[0] for i in batch]
            response = [i[1] for i in batch]
            candidates = [i[2] for i in batch]
            easy = [i[3] for i in batch]
            return {
                'context': context,
                'response': response,
                'candidates': candidates,
                'easy': easy,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            context, responses, label = batch[0]
            return {
                'label': label,
                'context': context,
                'responses': responses,
            }
