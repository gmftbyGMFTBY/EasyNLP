from header import *
from .utils import *


class SimCSEDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_simcse_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_dual_bert(path, lang=self.args['lang'], xlm=self.args['xlm'])
        self.data = []
        if args['mode'] == 'train':
            for label, context, response in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus([context])
                ids = self._length_limit(item['input_ids'][0])
                self.data.append({'ids': ids})
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids = self._length_limit(self.vocab.batch_encode_plus([batch[0][1]])['input_ids'][0])
                self.data.append({
                    'ids': ids,
                    'text': batch[0][1],
                })
                
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            bundle = self.data[i]
            ids = torch.LongTensor(bundle['ids'])
            return ids
        else:
            ids = torch.LongTensor(self.data[i]['ids'])
            text = self.data[i]['text']
            return ids, text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            ids_mask = self.generate_mask(ids)
            if torch.cuda.is_available():
                ids, ids_mask = ids.cuda(), ids_mask.cuda()
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
            }
        else:
            ids, text = batch[0]
            if torch.cuda.is_available():
                ids = ids.cuda()
            return {
                'ids': ids,
                'text': text,
            }
