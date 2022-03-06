from header import *
from .utils import *
from .randomaccess import *

class BERTDualFullWRInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')

        self.pp_path = f'{args["root_dir"]}/data/{args["dataset"]}/inference_wr.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = set()
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            line = ' '.join(utterances)
            items = jieba.analyse.extract_tags(line)
            self.data |= set(items)
        self.data = list(self.data)
        print(f'[!] collect {len(self.data)} words')
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = self.vocab.encode(self.data[i], add_special_tokens=False)
        return torch.LongTensor(ids), self.data[i]

    def save(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save the data into {self.pp_path}')
        
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
            'text': rid_text,
        }
