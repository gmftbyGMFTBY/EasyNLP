from model.utils import *

class BERTDualSCMHNMutualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNMutualEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=3)
        self.can_encoder = BertEmbedding(model=model, add_tokens=3)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.pad_token_id

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False, before_comp=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]

        ## combine context and response embeddings before comparison
        # rep_cid_backup: [B_r, B_c, E]
        rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        rep = rep_cid + rep_rid
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp, cid_rep, rid_rep

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E], rep_rid: [K, B_r, E]
        rep_rid = rid_rep_whole.permute(1, 0, 2)
        # rep_cid: [K, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        rep = rep_cid + rep_rid
        # rest: [K, B_r, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        if before_comp:
            return dp, dp2, cid_rep, rid_rep
        else:
            return dp, dp2

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        cid, cid_mask, rid, rid_mask = to_cuda(cid, cid_mask, rid, rid_mask)
        dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        dp = F.softmax(dp.squeeze(), dim=-1)
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        hn_rids = batch['hn_rids']
        hn_rids = list(chain(*hn_rids))
        rid = rid + hn_rids
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
        cid_mask = generate_mask(cid)
        rid_mask = generate_mask(rid)
        cid, rid, cid_mask, rid_mask = to_cuda(cid, rid, cid_mask ,rid_mask)
        batch_size = len(cid)

        # [B_c, B_r]
        loss = 0
        if self.args['before_comp']:
            dp, dp2, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, before_comp=True)
            # before comparsion, optimize the absolute semantic space
            dot_product = torch.matmul(cid_rep, rid_rep.t())
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        else:
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
