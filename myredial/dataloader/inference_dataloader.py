from header import *
from .utils import *

class BERTDualInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_inference_{args["tokenizer"]}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if args['model'] in ['dual-bert-gray']:
            responses = read_response_json_data(path, lang=self.args['lang'])
        else:
            responses = read_response_data(path, lang=self.args['lang'])
        self.data = []
        for res in tqdm(responses):
            item = self.vocab.encode(res)
            rids = self._length_limit(item)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def _length_limit(self, ids):
        if len(ids) > self.args['max_len']:
            ids = ids[:self.args['max_len']:]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

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
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            rid, rid_mask = rid.cuda(), rid_mask.cuda()
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }
