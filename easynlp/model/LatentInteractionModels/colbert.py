from model.utils import *


class ColBERTEncoder(nn.Module):
    
    def __init__(self, **args):
        super(ColBERTEncoder, self).__init__()
        model = args['pretrained_model']
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder = BertEmbedding(model=model)
        self.criterion = nn.CrossEntropyLoss()
        
    def _encode(self, cid, rid, nrid, cid_mask, rid_mask, nrid_mask):
        batch_size = len(cid)
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [B_c, S, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        nrid_rep = self.can_encoder(nrid, nrid_mask, hidden=True)  # [B_r, S, E]
        cid_rep, rid_rep, nrid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1), F.normalize(nrid_rep, dim=-1)
        # scores
        scores = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B*B, S_c, S_r]
        rid_length = rid_mask.sum(dim=-1).tolist()
        for idx in range(batch_size):
            scores[idx, :, rid_length[idx]:] = -np.inf
        scores = scores.max(dim=-1)[0]
        cid_length = cid_mask.sum(dim=-1).tolist()
        rest = []
        for idx in range(batch_size):
            s = scores[idx, :cid_length[idx]].sum()
            rest.append(s)
        rest = torch.stack(rest)
        # scores
        scores = torch.bmm(cid_rep, nrid_rep.permute(0, 2, 1))    # [B, S_c, S_r]
        rid_length = nrid_mask.sum(dim=-1).tolist()
        for idx in range(batch_size):
            scores[idx, :, rid_length[idx]:] = -np.inf
        scores = scores.max(dim=-1)[0]
        cid_length = cid_mask.sum(dim=-1).tolist()
        nrest = []
        for idx in range(batch_size):
            s = scores[idx, :cid_length[idx]].sum()
            nrest.append(s)
        nrest = torch.stack(nrest)
        return rest, nrest
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        batch_size, rseq_len = rid.size()
        
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True).expand(len(rid), -1, -1)    # [B_r, S, E]
        
        # rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        rid_rep = torch.zeros(batch_size, rseq_len, 768).cuda()

        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        scores = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B, S_c, S_r]
        rid_length = rid_mask.sum(dim=-1).tolist()
        for idx in range(batch_size):
            scores[idx, :, rid_length[idx]:] = -np.inf
        scores = scores.max(dim=-1)[0]
        rest = scores.sum(dim=-1)
        return rest
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        nrid = batch['nrids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        nrid_mask = batch['nrids_mask']

        batch_size = cid.shape[0]
        rest, nrest = self._encode(cid, rid, nrid, cid_mask, rid_mask, nrid_mask)

        input = torch.stack([rest, nrest]).t()    # [B, 2]
        target = torch.zeros(input.size(0)).cuda().to(torch.long)
        loss = self.criterion(input, target)

        # calculate accuracy
        acc = (input.max(dim=1)[1] == target).to(torch.float).mean().item()
        return loss, acc


class ColBERTV2Encoder(nn.Module):

    '''for dialog response selection task, we only test the cross-batch negative sampling'''
    
    def __init__(self, **args):
        super(ColBERTV2Encoder, self).__init__()
        model = args['pretrained_model']
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder = BertEmbedding(model=model)
        self.args = args
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        bsz_c, seqlen_c = cid.size()
        bsz_r, seqlen_r = rid.size()
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [B_c, S, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        # step 1: cross-batch gather
        cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)
        # step 2:
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())     # [B_c*S_c, B_r*S_r]
        # step 3: masking
        cid_mask_ = cid_mask.view(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask_ = rid_mask.view(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask_ * rid_mask_
        dp[mask == 0] = -np.inf
        # step 4: split
        dp = torch.stack(torch.split(dp, bsz_r, dim=-1), dim=-1)    # [B_c*S_c, B_r, S_r]
        dp = dp.max(dim=-1)[0]     # [B_c*S_c, B_r]
        # step 5: remask
        dp_ = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        # step 6:
        dp_ = torch.stack(torch.split(dp_, bsz_c, dim=-1), dim=-1)    # [B_r, B_c, S_c]
        dp_ = dp_.sum(dim=-1)    # [B_r, B_c]
        return dp_
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, cid_mask, rid, rid_mask)
        return dp.squeeze(dim=0)
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(rid)
        batch_size = cid.shape[0]
        dp = self._encode(cid, rid, cid_mask, rid_mask)
        dp /= self.args['temp']
        
        # constrastive loss
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
