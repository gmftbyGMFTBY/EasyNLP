from model.utils import *


class BERTDualSCMHNCompareEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNCompareEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEncoderWithEncHiddenModel.from_pretrained(
            pretrained_model_name_or_path=model,
            add_cross_attention=True,
            is_decoder=True,
        )
        # add the [EOS] token
        self.ctx_encoder.resize_token_embeddings(self.ctx_encoder.config.vocab_size + 1)
        self.can_encoder = BertEmbedding(model=model)
        self.topk = 1 + args['gray_cand_num']
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)    # [B_r, E]
        else:
            rid_rep = self.can_encoder(rid, rid_mask)    # [B_r*K, E]
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]
        rid_rep_ = rid_rep.unsqueeze(0).expand(len(rid_rep), -1, -1)    # [B_r, B_r(S), E]
        cid_rep = self.ctx_encoder(
            cid, cid_mask, encoder_hidden_states=rid_rep_,        
        ).last_hidden_state    # [B_c, S, E]
        cid_rep = cid_rep[:, 0, :]    # [B_c, E]

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B_c, B_r]
        if is_test:
            return dot_product

        # train the hard negative samples
        cid_rep = self.ctx_encoder(
            cid, cid_mask, encoder_hidden_states=rid_rep_whole
        ).last_hidden_state    # [B_c, S, E]
        cid_rep = cid_rep[:, 0, :]    # [B_c, E]
        hard_dot_product = torch.bmm(
            cid_rep.unsqueeze(1),      # [B_c, 1, E]
            rid_rep_whole.permute(0, 2, 1),    # [B_r, E, K]
        ).squeeze(1)    # [B_c, K]
        return dot_product, hard_dot_product

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        loss = 0
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc
