from model.utils import *

class BERTDualGradingEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualGradingEncoder, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']
        self.topk = args['topk_encoder']
        assert self.topk == 2
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoders = nn.ModuleList([
            BertEmbedding(model=model) for _ in range(self.topk)    
        ])
        self.args = args
        self.lamb = args['lambda']

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep_1 = self.can_encoders[0](ids, attn_mask)
        rid_rep_1 = self.can_encoders[1](ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep_1 = self.can_encoders[0](rid, rid_mask)
        rid_rep_2 = self.can_encoders[1](rid, rid_mask)
        dot_product_1 = torch.matmul(cid_rep, rid_rep_1.t()).squeeze(0)
        dot_product_2 = torch.matmul(cid_rep, rid_rep_2.t()).squeeze(0)

        dot_product = self.lamb * dot_product_1 + (1-self.lamb) * dot_product_2
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B, S]
        hrid = batch['hrids']    # [B*K, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hrid_mask = batch['hrids_mask']
        batch_size = len(cid)

        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_rep_1 = self.can_encoders[0](rid, rid_mask)     # [B, E]
        rid_rep_2 = self.can_encoders[1](hrid, hrid_mask)     # [B*K, E]
        rid_rep_21 = self.can_encoders[1](rid, rid_mask)     # [B, E]

        loss = 0
        # easy negative 
        dot_product = torch.matmul(cid_rep, rid_rep_1.t())     # [B, B]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
       
        # hard negative
        # [B, K, E] -> [B, K+1, E]
        rid_rep_2 = torch.stack(torch.split(rid_rep_2, self.topk))     # [B, K, E]
        rid_rep_2 = torch.cat([rid_rep_21.unsqueeze(1), rid_rep_2], dim=1)    # [B, K+1, E]
        # [B, 1, E] x [B, E, K] -> [B, 1, K] -> [B, K]
        dot_product = torch.bmm(cid_rep.unsqueeze(1), rid_rep_2.permute(0, 2, 1)).squeeze(1)     # [B, K+1]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), 0] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        return loss, acc
