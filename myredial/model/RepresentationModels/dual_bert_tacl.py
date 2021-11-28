from model.utils import *

class BERTDualTACLEncoder(nn.Module):

    '''three kinds of losses:
        1. sentence-level contrastive loss
        2. inner-sentence token-level contrastive loss (harmful for the performance)
        3. inner-pair token-level contrastive loss(from context to response)
        4. inner-pair token-level contrastive loss(from response to context)
    '''

    def __init__(self, **args):
        super(BERTDualTACLEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep_1 = self.ctx_encoder(cid, cid_mask)    # [B, S_c, E]
        cid_rep_2 = self.ctx_encoder(cid, cid_mask)    # [B, S_c, E]
        rid_rep_1 = self.can_encoder(rid, rid_mask)    # [B*K, S_r, E]
        rid_rep_2 = self.can_encoder(rid, rid_mask)    # [B*K, S_r, E]
        return (cid_rep_1, cid_rep_2), (rid_rep_1, rid_rep_2)

    def get_1st_loss(self, cid_rep, rid_rep):
        '''get the sentence-level contrastive loss:
            cid_rep/rid_rep: [B, S, E]
        '''
        c, r = cid_rep[:, 0, :], rid_rep[:, 0, :]
        bsz, _, _ = cid_rep.size()
        # constrastive loss
        dot_product = torch.matmul(c, r.t()) 
        mask = torch.zeros_like(dot_product)
        mask[range(bsz), range(bsz)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(bsz)).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc

    def get_2nd_loss(self, rep1, rep2):
        '''get the inner-sentence token-level contrastive loss; double dropout to get the different views
            rep: [B, S, E]
        '''
        bsz, seqlen, _ = rep1.size()
        score = torch.bmm(rep1, rep2.permute(0, 2, 1))    # [B, S, S]
        mask = torch.zeros_like(score)    # [B, S, S]
        mask[:, range(seqlen), range(seqlen)] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * mask
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        label = torch.LongTensor(torch.arange(seqlen)).cuda() 
        label = label.unsqueeze(0).expand(bsz, -1)    # [B, S]
        acc = (score.max(dim=-1)[1] == label).to(torch.float).mean().item()
        return loss, acc

    def get_3rd_loss(self, cid_rep, rid_rep, rid_mask):
        '''get the in-batch token-level contrastive loss
            cid_rep/rid_rep: [B, S, E]
            rid_mask: [B, S]
        '''
        bsz, ctx_seqlen, _ = cid_rep.size()
        _  , res_seqlen, _ = rid_rep.size()
        score = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B, S_c, S_r]
        score = score[:, 0, :]    # [B, S_r]; only use the [CLS] token embedding

        # ignore the pad token
        mask = torch.where(rid_mask == 0, torch.ones_like(rid_mask), torch.zeros_like(rid_mask)) 
        mask *= -1000
        score += mask

        label = torch.zeros_like(score)    # [B, S_r]
        label[range(bsz), 0] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * label
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        acc_num = (score.max(dim=-1)[1] == torch.zeros(bsz).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc
    
    def get_4th_loss(self, cid_rep, rid_rep, cid_mask):
        '''get the in-batch token-level contrastive loss
            cid_rep/rid_rep: [B, S, E]
        '''
        bsz, ctx_seqlen, _ = cid_rep.size()
        _  , res_seqlen, _ = rid_rep.size()
        score = torch.bmm(rid_rep, cid_rep.permute(0, 2, 1))    # [B, S_r, S_c]
        score = score[:, 0, :]    # [B, S_c]; only use the [CLS] token embedding
        
        # ignore the pad token
        mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask)) 
        mask *= -1000
        score += mask
        
        label = torch.zeros_like(score)    # [B, S_c]
        label[range(bsz), 0] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * label
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        acc_num = (score.max(dim=-1)[1] == torch.zeros(bsz).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc

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
        (cid_rep, _), (rid_rep, _) = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep, rid_rep = cid_rep[:, 0, :], rid_rep[:, 0, :]
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        (cid_rep_1, cid_rep_2), (rid_rep_1, rid_rep_2) = self._encode(cid, rid, cid_mask, rid_mask)
        # 1.
        loss1, acc1 = self.get_1st_loss(cid_rep_1, rid_rep_1)
        # 2. 
        loss2_1, acc_2_1 = self.get_2nd_loss(cid_rep_1, cid_rep_2)
        loss2_2, acc_2_2 = self.get_2nd_loss(rid_rep_1, rid_rep_2)
        acc2 = (acc_2_1 + acc_2_2) / 2
        loss2 = loss2_1 + loss2_2
        # 3. 
        loss3, acc3 = self.get_3rd_loss(cid_rep_1, rid_rep_1, rid_mask)
        # 4.
        loss4, acc4 = self.get_4th_loss(cid_rep_1, rid_rep_1, cid_mask)
        loss3 += loss4
        acc3 = (acc3 + acc4) / 2
        # return (loss1, loss2, loss3), (acc1, acc2, acc3)
        
        # discard the loss2, i.e., the inner-sentence contrastive loss
        return (loss1, torch.zeros_like(loss2), loss3), (acc1, 0., acc3)
        
        # discard the loss3 and loss4
        # return (loss1, loss2, torch.zeros_like(loss3)), (acc1, acc2,  0.)


class BERTDualTACLWithQRNegEncoder(nn.Module):

    '''three kinds of losses:
        1. sentence-level contrastive loss
        2. inner-pair token-level contrastive loss(from context to response)
        3. inner-pair token-level contrastive loss(from response to context)
        4. inner-pair token-level contrastive loss(from context to hard negative response[q-r recall] )
    '''

    def __init__(self, **args):
        super(BERTDualTACLWithQRNegEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.topk = args['gray_cand_num'] + 1
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep_1 = self.ctx_encoder(cid, cid_mask)    # [B, S_c, E]
        cid_rep_2 = self.ctx_encoder(cid, cid_mask)    # [B, S_c, E]
        rid_rep_1 = self.can_encoder(rid, rid_mask)    # [B*K, S_r, E]
        rid_rep_2 = self.can_encoder(rid, rid_mask)    # [B*K, S_r, E]
        return (cid_rep_1, cid_rep_2), (rid_rep_1, rid_rep_2)

    def get_1st_loss(self, cid_rep, rid_rep):
        '''get the sentence-level contrastive loss:
            cid_rep/rid_rep: [B, S, E]
        '''
        c, r = cid_rep[:, 0, :], rid_rep[:, 0, :]
        bsz, _, _ = cid_rep.size()
        # constrastive loss
        dot_product = torch.matmul(c, r.t()) 
        mask = torch.zeros_like(dot_product)
        mask[range(bsz), range(bsz)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(bsz)).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc

    def get_2nd_loss(self, cid_rep, rid_rep, rid_mask):
        '''get the in-batch token-level contrastive loss
            cid_rep/rid_rep: [B, S, E]
        '''
        bsz, ctx_seqlen, _ = cid_rep.size()
        _  , res_seqlen, _ = rid_rep.size()
        score = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B, S_c, S_r]
        score = score[:, 0, :]    # [B, S_r]; only use the [CLS] token embedding
        
        # ignore the pad token
        mask = torch.where(rid_mask == 0, torch.ones_like(rid_mask), torch.zeros_like(rid_mask)) 
        mask *= -1000
        score += mask

        label = torch.zeros_like(score)    # [B, S_r]
        label[range(bsz), 0] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * label
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        acc_num = (score.max(dim=-1)[1] == torch.zeros(bsz).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc
    
    def get_3rd_loss(self, cid_rep, rid_rep, cid_mask):
        '''get the in-batch token-level contrastive loss
            cid_rep/rid_rep: [B, S, E]
        '''
        bsz, ctx_seqlen, _ = cid_rep.size()
        _  , res_seqlen, _ = rid_rep.size()
        score = torch.bmm(rid_rep, cid_rep.permute(0, 2, 1))    # [B, S_r, S_c]
        score = score[:, 0, :]    # [B, S_c]; only use the [CLS] token embedding
        
        # ignore the pad token
        mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask)) 
        mask *= -1000
        score += mask
        
        label = torch.zeros_like(score)    # [B, S_c]
        label[range(bsz), 0] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * label
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        acc_num = (score.max(dim=-1)[1] == torch.zeros(bsz).cuda()).sum().item()
        acc = acc_num / bsz
        return loss, acc

    def get_4th_loss(self, cid_rep, rid_rep, rid_mask):
        '''get the inner-pair token-level contrastive loss
            cid_rep: [B, E]
            rid_rep: [B*K, S, E]
            rid_mask: [B*K, S]
        '''
        rid_rep = torch.stack(torch.split(rid_rep, self.topk))    # [B, K, S_r, E]
        cid_rep = cid_rep.unsqueeze(1)    # [B, 1, E]
        rest = []
        _, _, seqlen, _ = rid_rep.size()
        bsz, _, _ = cid_rep.size()
        for i in range(self.topk):
            rest_ = torch.bmm(cid_rep, rid_rep[:, i, :, :].permute(0, 2, 1))    # [B, 1, S_r]
            rest_ = rest_.squeeze(1)    # [B, S_r]
            rest.append(rest_)
        rest = torch.stack(rest)    # [K, B, S_r]
        rest = rest.permute(1, 0, 2)    # [B, K, S_r]
        rest = rest.reshape(len(rest), -1)    # [B, K*S_r]
        ## ignoration mask (padding mask and the groudtruth mask)
        # padding msak
        mask = torch.stack(torch.split(rid_mask, self.topk))    # [B, K, S_r]
        mask = mask.reshape(len(mask), -1)    # [B, K*S_r]
        mask = torch.where(mask == 0, torch.ones_like(mask), torch.zeros_like(mask)) 
        mask *= -1000
        # groundtruth mask
        mask[:, 1:seqlen] = -1000
        # CLS tokens are masked
        # for i in range(1, self.topk):
        #     mask[:, i*seqlen] = -1000

        rest += mask    # ignore the pad token
        label = torch.zeros_like(rest)
        label[range(bsz), 0] = 1.
        loss_ = F.log_softmax(rest, dim=-1) * label
        loss = (-loss_.sum(dim=-1)).mean()
        # acc
        acc = (rest.max(dim=-1)[1] == torch.zeros(bsz).cuda()).to(torch.float).mean().item()
        return loss, acc

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
        (cid_rep, _), (rid_rep, _) = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep, rid_rep = cid_rep[:, 0, :], rid_rep[:, 0, :]
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        (cid_rep_1, cid_rep_2), (rid_rep_1, rid_rep_2) = self._encode(cid, rid, cid_mask, rid_mask)
        # groundtruth response embeddings
        rid_rep_1_ = torch.stack(torch.split(rid_rep_1, self.topk))[:, 0, :, :]    # [B, S_r, E]
        rid_mask_ = torch.stack(torch.split(rid_mask, self.topk))[:, 0, :]    # [B, S]
        # 1.
        loss1, acc1 = self.get_1st_loss(cid_rep_1, rid_rep_1_)
        # 2. 
        loss2, acc2 = self.get_2nd_loss(cid_rep_1, rid_rep_1_, rid_mask_)
        # 3.
        loss3, acc3 = self.get_3rd_loss(cid_rep_1, rid_rep_1_, cid_mask)
        loss2 += loss3
        acc2 = (acc2 + acc3) / 2
        # 4.
        loss3, acc3 = self.get_4th_loss(cid_rep_1[:, 0, :], rid_rep_1, rid_mask)
        return (loss1, loss2, loss3), (acc1, acc2, acc3)
