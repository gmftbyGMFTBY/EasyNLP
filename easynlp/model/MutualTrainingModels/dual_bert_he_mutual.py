from model.utils import *


class BERTDualTripletMarginMutualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualTripletMarginMutualEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        score = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        return score
    
    def forward(self, batch):
        '''cid/rid/hrid: [B*K, S]'''
        cid = batch['ids']
        rid = batch['rids']
        hrid = batch['hrids']
        erid = batch['erids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hrid_mask = batch['hrids_mask']
        erid_mask = batch['erids_mask']

        # generate the representations
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        hrid_rep = self.can_encoder(hrid, hrid_mask)
        erid_rep = self.can_encoder(erid, erid_mask)
        
        # margin loss for topk hard negative samples
        loss = self.hard_criterion(cid_rep, rid_rep, hrid_rep)
        loss += self.easy_criterion(cid_rep, rid_rep, erid_rep)
        return loss
