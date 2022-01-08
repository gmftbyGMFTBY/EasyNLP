from model.utils import *


class BERTDualSCMTopKHNEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMTopKHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = TopKBertEmbedding(
            model=model, m=args['poly_m'], dropout=args['dropout']
        )
        self.can_encoder = TopKBertEmbedding(
            model=model, m=args['poly_m'], dropout=args['dropout']
        )
        # fusion layer
        fusion_layers = []
        for _ in range(args['poly_m']):
            decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
            fusion_layers.append(
                nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
            )
        self.fusion_encoder = nn.ModuleList(fusion_layers)
        self.squeeze = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*args['poly_m'], 768)
        )
        # parameters
        self.topk = 1 + args['gray_cand_num']
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_reps: [M, B_c, E]; cid_rep_whole: [B_c, S, E]
        cid_reps, cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        cid_rep_ = cid_reps.permute(1, 0, 2)    # [B_c, M, E]
        cid_rep_ = cid_reps.reshape(cid_rep_.size(0), -1)    # [B_c, M*E]
        cid_rep_ = cid_rep_.unsqueeze(1)    # [B_c, 1, M*E]
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        if is_test:
            # rid_rep: [M, B_r, E]
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            # rid_rep: [B_r*K, M, E]
            rid_rep = self.can_encoder(rid, rid_mask).permute(1, 0, 2)
            # rid_rep_whole: [M, B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk)).permute(2, 0, 1, 3)
            # rid_rep: [M, B_r, E]
            rid_rep = rid_rep_whole[:, :, 0, :]

        ## combine context and response embeddings before comparison
        rests = []
        for poly_i in range(self.args['poly_m']):
            # rep_rid: [B_r, B_c, E]
            rep_rid = rid_rep[poly_i].unsqueeze(1).expand(-1, cid_size, -1)
            # rep_cid: [B_r, B_c, E]
            rep_cid = cid_reps[poly_i].unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
            # rest: [B_r, B_c, E]
            rest = self.fusion_encoder[poly_i](
                rep, 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rests.append(rest)
        # [M, B_r, B_c, E]
        rests = torch.stack(rests)
        # [B_r, B_c, M, E]
        rests = rests.permute(1, 2, 0, 3)
        # [B_r, B_c, E*M]
        rests = rests.reshape(rests.size(0), rests.size(1), -1)
        rests = self.squeeze(rests)
        # rest: [B_c, E*M, B_r]
        rests = rests.permute(1, 2, 0)
        cid_rep_ = self.squeeze(cid_rep_)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rests).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rep_rid: [M, B_r, K, E] -> [M, K, B_r, E]
        rep_rid = rid_rep_whole.permute(0, 2, 1, 3)
        # rep_cid: [M, B_c, E] -> [M, K, B_c, E]
        rep_cid = cid_reps.unsqueeze(1).expand(-1, rep_rid.size(1), -1, -1)
        # rep: [M, K, B, E]
        rep = rep_cid + rep_rid
        rests = []
        for poly_i in range(self.args['poly_m']):
            # rest: [K, B_r, E]
            rest = self.fusion_encoder[poly_i](
                rep[poly_i], 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rests.append(rest)
        # [M, K, B_r, E]
        rests = torch.stack(rests)
        # [K, B_r, M, E]
        rests = rests.permute(1, 2, 0, 3)
        # [K, B_r, M*E]
        rests = rests.reshape(rests.size(0), rests.size(1), -1)
        rests = self.squeeze(rests)
        # [B_r, E, K]
        rests = rests.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rests).squeeze(1)
        return dp, dp2

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
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc
