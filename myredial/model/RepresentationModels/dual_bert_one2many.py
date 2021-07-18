from model.utils import *

class BERTDualO2MEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualO2MEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['topk_encoder']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoders = nn.ModuleList([
            BertEmbedding(model=model) for _ in range(self.topk) 
        ])

    def _encode(self, cid, rid, cid_mask, rid_mask, test=False):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_reps = []
        # num = 1 if test else self.topk
        num = self.topk
        for idx in range(num):
            rid_rep = self.can_encoders[idx](rid[idx], rid_mask[idx])
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_reps = []
        for idx in range(self.topk):
            rid_rep = self.can_encoders[idx](ids, attn_mask)
            rid_reps.append(rid_rep)
        # K*[B, E]
        rid_rep = torch.cat(rid_reps)    # [B*K, E]
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
        cid_rep, rid_reps = self._encode(cid, [rid] * self.topk, cid_mask, [rid_mask] * self.topk, test=True)
        dot_products = []
        for rid_rep in rid_reps:
            dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)    # [B]
            dot_products.append(dot_product)
        dot_products = torch.stack(dot_products)
        score = dot_products.max(dim=0)[0]
        return score
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = len(cid)
        cid_rep, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)
        loss = 0
        dot_products = []
        for i, rid_rep in enumerate(rid_reps):
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
            # constrastive loss
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product)
        loss /= len(rid_reps)

        # acc
        acc_num = 0
        for dot_product_ in dot_products:
            acc_num += (F.softmax(dot_product_, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / self.topk
        return loss, acc
