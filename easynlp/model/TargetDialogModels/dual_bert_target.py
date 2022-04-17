from model.utils import *


class BERTDualTargetEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualTargetEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.args = args

    @torch.no_grad()
    def get_ctx_embedding(self, ids, attn_mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, attn_mask)    # [B, E]
        cid_rep = F.normalize(cid_rep, dim=-1)
        return cid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = F.normalize(rid_rep, dim=-1)
        return rid_rep

    def _encode(self, cids, rid, cids_mask, rid_mask):
        cid_rep = self.ctx_encoder(cids, cids_mask)    # [B, S, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, E]
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        cid = batch['ids'].unsqueeze(0)
        rid = batch['rids']
        cid_mask = torch.ones_like(cid)
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
