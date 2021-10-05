from model.utils import *


class BERTDualEncoderBCE(nn.Module):

    '''Dual bert with hard negative samples, but the bce loss is used instead of the contrastive loss'''

    def __init__(self, **args):
        super(BERTDualEncoderBCE, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Linear(768, 1)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        bsz_c, bsz_r = len(cid_rep), len(rid_rep)
        cid_rep = cid_rep.unsqueeze(1).expand(-1, bsz_r, -1)    # [B_c, B_r, E]
        rid_rep = rid_rep.unsqueeze(0).expand(bsz_c, -1, -1)    # [B_c, B_r, E]
        rep = self.fusion_head(torch.cat([cid_rep, rid_rep], dim=-1)).squeeze(-1)    # [B_c, B_r]
        return rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask).squeeze(0)    # [B_r]
        dot_product = torch.sigmoid(dot_product)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        batch_size = len(cid)

        # BCE loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        dot_product, mask = dot_product.view(-1), mask.view(-1)
        a = dot_product.view(-1)
        b = mask.view(-1)

        # random shuffle
        random_idx = list(range(len(a)))
        random.shuffle(random_idx)
        a = torch.stack([a[i] for i in random_idx])
        b = torch.stack([b[i] for i in random_idx])
        loss = self.criterion(a, b)

        # acc
        acc = ((torch.sigmoid(dot_product).view(-1) > 0.5).float() == mask.view(-1)).float().mean().item()

        return loss, acc


class BERTDualEncoderTripletMargin(nn.Module):

    def __init__(self, **args):
        super(BERTDualEncoderTripletMargin, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.margin = args['margin']
        self.topk = args['gray_cand_num']
        self.criterion = nn.TripletMarginWithDistanceLoss(
            margin=self.margin,
            reduction='sum',
            distance_function=cosine_distance,
        )

    def _fast_n_pair_loss(self, cid_rep, rid_rep):
        '''L(x,x+,\{x_i\}_{i=1}^{N-1})=\log(1+\sum_{i=1}^{N-1}exp(x^Tx_i-x^Tx+))'''
        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product -= dot_product.diag().unsqueeze(1)
        dot_product[range(len(cid_rep)), range(len(cid_rep))] = -1e3
        dot_product = torch.exp(dot_product)
        dot_product = dot_product.sum(dim=1) + 1    # [B]
        loss = torch.log(dot_product).sum()
        return loss

    def _fast_triplet_margin_cosine_distance_loss(self, cid_rep, rid_rep):
        '''cid_rep/rid_rep: [B, E]; reduction is the `sum`'''
        cosine_sim = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        cosine_sim = (1 + cosine_sim) / 2    # nonnegative real-value number, range from 0 to 1
        # triplet with margin loss
        cosine_sim = self.margin + cosine_sim - cosine_sim.diag().unsqueeze(1)
        cosine_sim = torch.where(cosine_sim > 0, cosine_sim, torch.zeros_like(cosine_sim))
        # ignore the ground-truth
        cosine_sim[range(len(cid_rep)), range(len(cid_rep))] = 0.
        # only topk negative will be optimized
        # values = torch.topk(cosine_sim, self.topk, dim=-1)[0]    # [B, K]
        # valid_num = len(cosine_sim.nonzero())
        # loss = cosine_sim.sum() / valid_num
        loss = cosine_sim.sum()
        return loss

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity needs the normalization
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)    # [B]
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        
        # get acc
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        loss = self._fast_triplet_margin_cosine_distance_loss(cid_rep, rid_rep)
        # loss = self._fast_n_pair_loss(cid_rep, rid_rep)
        return loss, acc
        
        # dot_product[range(batch_size), range(batch_size)] = -1e3
        # hn_index = torch.topk(dot_product, self.topk, dim=-1)[1].tolist()

        # margin loss for topk hard negative samples
        # anchor_reps, rid_reps, nid_reps = [], [], []
        # for idx in range(batch_size):
        #     hn = hn_index[idx]
        #     for i in hn:
        #         anchor_reps.append(cid_rep[idx])
        #         rid_reps.append(rid_rep[idx])
        #         nid_reps.append(rid_rep[i])
        # anchor_reps = torch.stack(anchor_reps)
        # rid_reps = torch.stack(rid_reps)
        # nid_reps = torch.stack(nid_reps)
        # loss = self.criterion(anchor_reps, nid_reps, rid_reps)
        # return loss, acc
