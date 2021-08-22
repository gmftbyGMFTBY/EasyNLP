from model.utils import *

class BERTDualHNEncoder(nn.Module):

    '''Dual bert with hard negative samples'''

    def __init__(self, **args):
        super(BERTDualHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
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
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHNHierEncoder(nn.Module):

    '''Dual bert with hard negative samples'''

    def __init__(self, **args):
        super(BERTDualHNHierEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num']
        p = args['dropout']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.fg_head_ctx = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.fg_head_res = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep_ = self.fg_head_res(rid_rep)
        rid_rep = torch.cat([rid_rep, rid_rep_], dim=1)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep_ = self.fg_head_res(cid_rep)
        cid_rep = torch.cat([cid_rep, cid_rep_], dim=1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep_ = self.fg_head_ctx(cid_rep)
        rid_rep_ = self.fg_head_res(rid_rep)
        cid_rep = torch.cat([cid_rep, cid_rep_], dim=1)
        rid_rep = torch.cat([rid_rep, rid_rep_], dim=1)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B, S]
        hrid = batch['hrids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hrid_mask = batch['hrids_mask']

        # first layer: easy
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        batch_size = len(cid_rep)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # second layer: hard
        cid_rep = self.fg_head_ctx(cid_rep)    # [B, E]
        rid_rep = self.fg_head_res(rid_rep)    # [B, E]
        hrid_rep = self.can_encoder(hrid, hrid_mask)    # [B*M, E]
        hrid_rep = self.fg_head_res(hrid_rep)    # [B*M, E]
        hrid_rep = torch.stack(torch.split(hrid_rep, self.topk))    # [B, M, E]
        hrid_rep = torch.cat([rid_rep.unsqueeze(1), hrid_rep], dim=1)    # [B, M+1, E]
        dot_product = torch.bmm(cid_rep.unsqueeze(1), hrid_rep.permute(0, 2, 1)).squeeze(1)    # [B, M+1]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), 0] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.zeros(batch_size).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc


class BERTDualHNPosEncoder(nn.Module):

    '''Dual bert with hard negative samples, and the position weight for the context'''

    def __init__(self, **args):
        super(BERTDualHNPosEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, cid_pos):
        cid_reps = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask, cid_pos):
        cid_reps = self.ctx_encoder(ids, attn_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w'].unsqueeze(0)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos) 
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid_rep), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid_rep), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHNContextEncoder(nn.Module):

    '''Dual bert with hard negative samples and hard negative context'''

    def __init__(self, **args):
        super(BERTDualHNContextEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
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
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hcid = batch['hids']
        hcid_mask = batch['hids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep_ = self.ctx_encoder(cid, cid_mask)
        hcid_rep = self.ctx_encoder(hcid, hcid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # constrastive loss
        dp = torch.matmul(cid_rep, torch.cat([cid_rep_, hcid_rep], dim=0).t())
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
