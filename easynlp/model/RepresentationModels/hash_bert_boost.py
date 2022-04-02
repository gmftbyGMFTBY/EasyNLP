from model.utils import *
from model.LanguageModels import *
from .hash_bert import *

class HashBERTDualBoostEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualBoostEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.temp = args['temp']
        self.select_topk = args['select_topk']

        # base model
        base_args = deepcopy(args)
        base_args['hash_code_size'] = args['base_hash_code_size']
        self.base_model = HashBERTDualEncoder(**base_args)
        inpt_size = 768
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.criterion = nn.MarginRankingLoss(margin=self.hash_code_size)

    def compact_binary_vectors(self, ids):
        # ids: [B, D]
        ids = ids.cpu().numpy().astype('int')
        ids = np.split(ids, int(ids.shape[-1]/8), axis=-1)
        ids = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in ids]    
            ).transpose().astype('uint8')
        )
        return ids

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, ids_mask)
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        hash_code = torch.from_numpy(hash_code)
        return hash_code
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, ids_mask)
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        return hash_code

    @torch.no_grad()
    def predict(self, batch):
        self.base_model.eval()
        self.ctx_hash_encoder.eval()
        self.can_hash_encoder.eval()

        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = torch.ones_like(cid), batch['rids_mask']

        cid_rep = self.base_model.ctx_encoder(cid, cid_mask)
        rid_rep = self.base_model.can_encoder(rid, rid_mask)
        ctx_hash_code_base = torch.sign(self.base_model.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code_base = torch.sign(self.base_model.can_hash_encoder(rid_rep))    # [B, Hash]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        ctx_hash_code = torch.cat([ctx_hash_code, ctx_hash_code_base], dim=-1)
        can_hash_code = torch.cat([can_hash_code, can_hash_code_base], dim=-1)
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t()).squeeze(0)    # [B]
        distance = -0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        batch_size = len(cid)

        with torch.no_grad():
            cid_rep = self.base_model.ctx_encoder(cid, cid_mask)
            rid_rep = self.base_model.can_encoder(rid, rid_mask)
            ctx_hash_code = torch.sign(self.base_model.ctx_hash_encoder(cid_rep))
            can_hash_code = torch.sign(self.base_model.can_hash_encoder(rid_rep))
            matrix = torch.matmul(ctx_hash_code, can_hash_code.t())    # [B, B]
            distance = - 0.5 * (self.base_model.hash_code_size - matrix)    # [B, B]
            topk_dis = distance.topk(self.select_topk)[1]

            ctx_reps, pos_reps, neg_reps = [], [] ,[]
            for idx in range(batch_size):
                topk_dis_ = topk_dis[idx]    # [K]
                neg_index = []
                for item in topk_dis_:
                    if item == idx:
                        break
                    neg_index.append(item)
                if len(neg_index) > 0:
                    ctx_reps.extend([cid_rep[idx] for _ in range(len(neg_index))])
                    pos_reps.extend([rid_rep[idx] for _ in range(len(neg_index))])
                    neg_reps.extend([rid_rep[i] for i in neg_index])
            ctx_reps = torch.stack(ctx_reps)
            pos_reps = torch.stack(pos_reps)
            neg_reps = torch.stack(neg_reps)

        if len(ctx_reps) == 0:
            hash_loss = torch.tensor(0.).cuda()
            return hash_loss, 0.
        ctx_hash_code = self.ctx_hash_encoder(ctx_reps) 
        pos_hash_code = self.can_hash_encoder(pos_reps)   
        neg_hash_code = self.can_hash_encoder(neg_reps)    
        pos_matrix = torch.einsum('ij,ij->i', ctx_hash_code, pos_hash_code)
        neg_matrix = torch.einsum('ij,ij->i', ctx_hash_code, neg_hash_code)
        hash_loss = self.criterion(pos_matrix, neg_matrix, torch.ones_like(pos_matrix))

        # ===== calculate hamming distance for accuracy ===== #
        acc = (pos_matrix > neg_matrix).to(torch.float).mean().item()
        return hash_loss, acc
