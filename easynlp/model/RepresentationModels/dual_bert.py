from model.utils import *

class BERTDualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualEncoder, self).__init__()

        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # rid_rep = torch.randn(10, 768).cuda()
        # cosine similarity
        # cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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


class BERTDualWithMarginEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualWithMarginEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.margin = args['margin']
        self.topk = args['gray_cand_num']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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
        dot_product *= self.temp
        return dot_product
    
    def _fast_triplet_margin_cosine_distance_loss(self, cid_rep, rid_rep):
        '''cid_rep/rid_rep: [B, E]; reduction is the `sum`'''
        cosine_sim = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        # nonnegative real-value number, range from 0 to 1
        cosine_sim = (1 + cosine_sim) / 2
        # triplet with margin loss
        cosine_sim = self.margin + cosine_sim - cosine_sim.diag().unsqueeze(1)
        cosine_sim = torch.where(cosine_sim > 0, cosine_sim, torch.zeros_like(cosine_sim))
        # ignore the ground-truth
        cosine_sim[range(len(cid_rep)), range(len(cid_rep))] = 0.
        # only topk negative will be optimized
        values = torch.topk(cosine_sim, self.topk, dim=-1)[0]    # [B, K]
        loss = values.mean()
        return loss

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # triplet margin loss
        loss += self._fast_triplet_margin_cosine_distance_loss(cid_rep, rid_rep)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualNHPEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualNHPEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity
        # cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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
        rid_1 = batch['rids_1']
        rid_2 = batch['rids_2']
        cid_mask = batch['ids_mask']
        rid_1_mask = batch['rids_1_mask']
        rid_2_mask = batch['rids_2_mask']
        batch_size = len(cid)

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_1_rep = self.can_encoder(rid_1, rid_1_mask)
        rid_2_rep = self.can_encoder(rid_2, rid_2_mask)

        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_1_rep.t()) 
        dot_product /= self.temp

        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # 
        dot_product = torch.matmul(cid_rep, rid_2_rep.t()) 
        dot_product /= self.temp

        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        return loss, acc


class BERTDualDCLEncoder(nn.Module):

    '''refer to the paper: decoupled  contrastive learning'''

    def __init__(self, **args):
        super(BERTDualDCLEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
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

    def log_sum_exp_trick(self, data):
        # data: [B, B-1]
        a = data.max(dim=1)[0]    # [B]
        data = torch.exp(data - a.unsqueeze(dim=1))
        data = a + torch.log(data.sum(dim=-1))
        return data
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        batch_size = len(cid_rep)

        # DCL: decoupling contrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp
        
        ## acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        ## term_1
        term_1 = -dot_product.diag()    # [B]
        ## term_2
        index = torch.ones_like(dot_product)
        index[range(batch_size), range(batch_size)] = 0.
        index = index.to(torch.bool)
        dot_product = dot_product[index].reshape(batch_size, batch_size-1)    # [B, B-1]
        term_2 = self.log_sum_exp_trick(dot_product)    # [B]

        ## loss
        loss_ = term_1 + term_2
        loss = loss_.mean()
        return loss, acc


class BERTDualWideDPEncoder(nn.Module):

    '''wide dot production'''

    def __init__(self, **args):
        super(BERTDualWideDPEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, S_c, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, S_r, E]
        ctx_batch_size, res_batch_size = len(cid_rep), len(rid_rep)

        # replace the [PAD] token with torch.zeros
        # mask: [B, S, E]
        cid_mask_ = cid_mask.unsqueeze(-1).expand(-1, -1, 768)
        rid_mask_ = rid_mask.unsqueeze(-1).expand(-1, -1, 768)
        cid_mask_ = torch.where(cid_mask_ == 0, torch.ones_like(cid_mask_), torch.zeros_like(cid_mask_))
        rid_mask_ = torch.where(rid_mask_ == 0, torch.ones_like(rid_mask_), torch.zeros_like(rid_mask_))

        cid_rep.masked_fill_(cid_mask_.to(torch.bool), torch.tensor(0.))
        rid_rep.masked_fill_(rid_mask_.to(torch.bool), torch.tensor(-1.))
        
        # [B_c*B_r, S_c, E]
        cid_rep = torch.stack(list(chain(*[[item]*res_batch_size for item in cid_rep])))
        # [B_r*B_c, S_r, E]
        rid_rep = torch.cat([rid_rep] * ctx_batch_size)
        # [B_c*B_r, S_c, S_r]
        dot_product = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))
        # [B_c, B_r, S_c, S_r]
        dot_product = torch.stack(torch.split(dot_product, res_batch_size))
        dot_product = dot_product.sum(dim=-1).sum(dim=-1)    # [B_c, B_r]
        # divide the effective number of the tokens for average
        cid_effective_num = cid_mask.sum(dim=-1).unsqueeze(-1).expand(-1, res_batch_size)
        rid_effective_num = rid_mask.sum(dim=-1).unsqueeze(0).expand(ctx_batch_size, -1)
        effective_num = cid_effective_num * rid_effective_num

        dot_product /= self.temp
        dot_product /= effective_num
        return dot_product

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
        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        return dot_product.squeeze(0)
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
