from header import *
from .utils import *


class BERTDualFullWithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.data = []
        if self.args['mode'] == 'train':

            path_head = os.path.splitext(path)[0]
            self.paths = [f"{path_head}.splita{i}" for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
            for path in self.paths:
                # clear the self.data and load the data from path
                data, responses = read_json_data(
                    path, lang=self.args['lang']
                )
                for context, response, candidates in tqdm(data):
                    context = ' [SEP] '.join(context).strip()
                    if len(candidates) < 10:
                        candidates += random.sample(responses, 10-len(candidates))
                    else:
                        candidates = candidates[:10]
                    self.data.append({
                        'context': context,
                        'responses': [response] + candidates,
                    })
        else:
            data, responses = read_json_data(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                self.data.append({
                    'label': [1] + [0] * 9,
                    'context': context,
                    'responses': [response] + candidates,
                })

    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids
                
    def __len__(self):
        # Lol. Shit Code
        return 37373824

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            context = bundle['context']
            responses = [bundle['responses'][0]] + random.sample(bundle['responses'][1:], self.args['gray_cand_num'])
            return context, responses
        else:
            context = bundle['context']
            responses = bundle['responses']
            return context, responses, bundle['label']

    def save(self):
        pass
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            context = [i[0] for i in batch]
            responses = []
            for i in batch:
                responses.extend(i[1])
            return {
                'context': context, 
                'responses': responses, 
            }
        else:
            assert len(batch) == 1
            batch = batch[0]
            context, responses, label = batch[0], batch[1], batch[2]
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                label = label.cuda()
            return {
                'context': context, 
                'responses': responses, 
                'label': label
            }
