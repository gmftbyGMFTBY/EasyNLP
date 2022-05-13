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
        
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        # rid_rep = torch.zeros(batch_size, rseq_len, 768).cuda()

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
        self.can_encoder = BertEmbedding(model=model, add_tokens=3)
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=3)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.pad_token_id
        self.args = args
        
    def _encode(self, cid, rid, cid_mask, rid_mask, test=True):
        bsz_c, seqlen_c = cid.size()
        bsz_r, seqlen_r = rid.size()
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [B_c, S, E]
        # rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        rid_rep = torch.zeros(len(rid), len(rid[0]), 768).cuda()
        cid_rep = F.normalize(cid_rep, dim=-1)
        rid_rep = F.normalize(rid_rep, dim=-1)
        # step 1: cross-batch gather
        # if test:
        #     cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)
        #     cid_mask, rid_mask = distributed_collect(cid_mask, rid_mask)
        #     bsz_c *= torch.distributed.get_world_size()
        #     bsz_r *= torch.distributed.get_world_size()
        # step 2:
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())     # [B_c*S_c, B_r*S_r]
        # step 3: masking
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf
        # step 4: split and maximum
        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)    # [B_c*S_c, B_r, S_r]
        dp = dp.max(dim=-1)[0]     # [B_c*S_c, B_r]
        # step 5: remask
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        # step 6: sum
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, test=False)
        return dp.squeeze(dim=0)
        
    def forward(self, batch):
        # cid = batch['ids']
        # rid = batch['rids']
        # cid_mask = batch['ids_mask']
        # rid_mask = batch['rids_mask']

        cid = batch['ids']
        rid = batch['rids']
        # hn_rid = batch['hn_rids']
        # hn_rid = list(chain(*hn_rid))
        # rid += hn_rid
        cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        cid_mask = generate_mask(cid)
        rid_mask = generate_mask(rid)

        dp = self._encode(cid, rid, cid_mask, rid_mask)
        dp /= self.args['temp']
        batch_size = len(dp)
        
        # constrastive loss
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc


class ColBERTV2TaCLEncoder(nn.Module):

    def __init__(self, **args):
        super(ColBERTV2TaCLEncoder, self).__init__()
        model = args['pretrained_model']
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder = BertEmbedding(model=model)
        self.args = args

    def get_tacl_loss(self, hidden):
        bsz, seqlen, _ = hidden.size()
        dp = torch.matmul(hidden, hidden.permute(0, 2, 1))    # [B_c, S, S]
        gold_score = torch.diagonal(dp, offset=0, dim1=1, dim2=2)    # [B_c, S]
        gold_score = gold_score.unsqueeze(dim=-1)    # [B_c, S, 1]
        difference = gold_score - dp
        loss = self.args['margin'] - difference
        loss[:, range(seqlen), range(seqlen)] = 0
        loss = F.relu(loss)    # [B_c, S, S]
        return loss.sum(dim=-1).sum(dim=-1).mean()
        
    def _encode(self, cid, rid, cid_mask, rid_mask, test=False):
        bsz_c, seqlen_c = cid.size()
        bsz_r, seqlen_r = rid.size()
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [B_c, S, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B_r, S, E]
        cid_rep = F.normalize(cid_rep, dim=-1)
        rid_rep = F.normalize(rid_rep, dim=-1)

        # tacl loss
        if test is False:
            tacl_loss = self.get_tacl_loss(cid_rep)
            tacl_loss += self.get_tacl_loss(rid_rep)
            # step 1: cross-batch gather
            # cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)
            # cid_mask, rid_mask = distributed_collect(cid_mask, rid_mask)
            # bsz_c *= torch.distributed.get_world_size()
            # bsz_r *= torch.distributed.get_world_size()
        # step 2:
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())     # [B_c*S_c, B_r*S_r]
        # step 3: masking
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf
        # step 4: split and maximum
        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)    # [B_c*S_c, B_r, S_r]
        dp = dp.max(dim=-1)[0]     # [B_c*S_c, B_r]
        # step 5: remask
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        # step 6: sum
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]

        # tacl loss
        if test:
            return dp
        else:
            return dp, tacl_loss
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, test=True)
        return dp.squeeze(dim=0)
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        dp, tacl_loss = self._encode(cid, rid, cid_mask, rid_mask)
        dp /= self.args['temp']
        batch_size = len(dp)
        
        # constrastive loss
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, tacl_loss, acc
