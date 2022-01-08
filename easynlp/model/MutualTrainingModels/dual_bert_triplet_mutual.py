from model.utils import *


class BERTDualTripletMarginMutualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualTripletMarginMutualEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.temp = args['temp']
        self.hard_margin = args['training_hard_margin']
        self.easy_margin = args['training_easy_margin']
        # for debug
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.hard_criterion = nn.TripletMarginWithDistanceLoss(
            margin=self.hard_margin,
            reduction='sum',
            distance_function=cosine_distance,
        )
        self.easy_criterion = nn.TripletMarginWithDistanceLoss(
            margin=self.easy_margin,
            reduction='sum',
            distance_function=cosine_distance,
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        score = F.cosine_similarity(cid_rep, rid_rep, dim=-1)
        return score
    
    def forward(self, batch, loss_type='triplet-loss'):
        if loss_type == 'triplet-loss':
            cid = batch['ids']
            rid = batch['rids']
            hrid = batch['hrids']
            erid = batch['erids']
            cid_mask = batch['ids_mask']
            rid_mask = batch['rids_mask']
            hrid_mask = batch['hrids_mask']
            erid_mask = batch['erids_mask']
            
            # generate the representation
            cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
            hrid_rep = self.can_encoder(hrid, hrid_mask)
            erid_rep = self.can_encoder(erid, erid_mask)
            hrid_rep, erid_rep = F.normalize(hrid_rep), F.normalize(erid_rep)
            # margin loss for topk hard negative samples
            # update the cid_rep and rid_rep with the same shape of the hrid_rep and erid_rep
            loss = self.hard_criterion(cid_rep, rid_rep, hrid_rep)
            loss += self.easy_criterion(cid_rep, rid_rep, erid_rep)
        else:
            # hold the semantic space (contrastive loss)
            dp_cid_rep, dp_rid_rep = self._encode(
                batch['ids'], batch['rids'],
                batch['ids_mask'], batch['rids_mask']
            )
            dp = torch.matmul(dp_cid_rep, dp_rid_rep.t())
            dp /= self.temp
            batch_size = len(dp_cid_rep)
            mask = torch.zeros_like(dp)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp, dim=-1) * mask
            loss = (-loss_.sum(dim=1)).mean()
        return loss
