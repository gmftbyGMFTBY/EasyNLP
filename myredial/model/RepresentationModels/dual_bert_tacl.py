from model.utils import *

class BERTDualTACLEncoder(nn.Module):

    '''three kinds of losses:
        1. sentence-level contrastive loss
        2. inner-sentence token-level contrastive loss
        3. inner_pair token-level contrastive loss
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
        rid_rep_1 = self.can_encoder(rid, rid_mask)    # [B, S_r, E]
        rid_rep_2 = self.can_encoder(rid, rid_mask)    # [B, S_r, E]
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

    def get_3rd_loss(self, cid_rep, rid_rep):
        '''get the in-batch token-level contrastive loss
            cid_rep/rid_rep: [B, S, E]
        '''
        bsz, ctx_seqlen, _ = cid_rep.size()
        _  , res_seqlen, _ = rid_rep.size()
        score = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B, S_c, S_r]
        score = score[:, 0, :]    # [B, S_r]; only use the [CLS] token embedding
        mask = torch.zeros_like(score)    # [B, S_r]
        mask[range(bsz), 0] = 1. 
        loss_ = F.log_softmax(score, dim=-1) * mask
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
        loss3, acc3 = self.get_3rd_loss(cid_rep_1, rid_rep_1)
        return (loss1, loss2, loss3), (acc1, acc2, acc3)
