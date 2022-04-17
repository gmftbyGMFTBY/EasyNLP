from model.utils import *

'''add the pair-wise comparison'''

class BERTDualSCMCompEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMCompEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r, E]
        # rid_rep = self.can_encoder(rid, rid_mask)
        rid_rep = torch.zeros(rid_size, 768).cuda()

        # cid_rep_mt, rid_rep_mt = self.convert_ctx(cid_rep), self.convert_res(rid_rep)
        cid_rep_mt, rid_rep_mt = cid_rep.clone(), rid_rep.clone()

        ## combine context and response embeddings before comparison
        # cid_rep: [B_r, B_c, E]
        cid_rep = cid_rep.unsqueeze(0).expand(rid_size, -1, -1)
        # rid_rep: [B_r, B_c, E]
        rid_rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        rep = rid_rep + cid_rep
        # rep: [B_r, B_c, 2*E]

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
        dp_dp = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp_dp, cid_rep_mt, rid_rep_mt

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.eval()
        rest = self.can_encoder(ids, ids_mask)
        rest = self.convert_res(rest)
        return rest
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.eval()
        rest = self.ctx_encoder(ids, ids_mask)
        rest = self.convert_ctx(rest)
        return rest

    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dp, cid_rep_mt, rid_rep_mt = self._encode(cid, rid, cid_mask, rid_mask)
        loss, loss_margin = 0, 0
        # multi-task: recall training
        if self.args['coarse_recall_loss']:
            dp_mt = torch.matmul(cid_rep_mt, rid_rep_mt.t())
            mask = torch.zeros_like(dp_mt)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp_mt, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()

        # multi-task: rerank training (ranking loss)
        ## dp: [B_c, B_r]
        # gold_score = torch.diagonal(dp).unsqueeze(dim=-1)    # [B_c, 1]
        # difference = gold_score - dp    # [B_c, B_r]
        # loss_matrix = torch.clamp(self.args['margin'] - difference, min=0.)   # [B_c, B_r]
        # loss_margin += loss_matrix.mean()
        
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        
        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMHNCompEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNCompEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        self.args = args

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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        # dp = torch.matmul(cid_rep, rid_rep.t())
        dp = F.softmax(dp.squeeze(), dim=-1)
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
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
