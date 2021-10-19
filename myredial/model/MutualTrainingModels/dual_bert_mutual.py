from model.utils import *

class BERTDualMutualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualMutualEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = F.normalize(rid_rep)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep = F.normalize(cid_rep)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask =  batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        score = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        return score

    def forward(self, batch):
        '''MSELoss'''
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        soft_label = batch['soft_label']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        criterion = nn.MSELoss()
        cosine_sim = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        loss = criterion(cosine_sim, soft_label)

        return loss
