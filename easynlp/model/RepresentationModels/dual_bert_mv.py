from model.utils import *

class BERTDualMVEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualMVEncoder, self).__init__()

        model = args['pretrained_model']
        self.temp = args['temp']
        self.mv = args['mv_num']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)
        r_bsz, r_seqlen, esz = rid_rep.size()
        c_bsz, c_seqlen, esz = cid_rep.size()
        assert r_seqlen >= self.mv, f"expected mv num: {self.mv}, but get {seqlen}"
        assert c_seqlen >= self.mv, f"expected mv num: {self.mv}, but get {seqlen}"
        cid_rep = cid_rep[:, :self.mv, :]
        rid_rep = rid_rep[:, :self.mv, :]
        cid_rep = cid_rep.reshape(c_bsz, self.mv*768)
        rid_rep = rid_rep.reshape(r_bsz, self.mv*768)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = F.normalize(rid_rep, dim=-1)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep = F.normalize(cid_rep, dim=-1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product = (dot_product + 1)/2
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
