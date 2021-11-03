from model.utils import *


class BERTDualSCMEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.fusion_encoder = BertEmbeddingWithEncoderHidden(model=model)

        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [B_c, S, E]
        cid_rep_ = cid_rep[:, 0, :]    # [B_c, E]
        rid_rep = self.can_encoder(rid, rid_mask)     # [B_r, E]
        rid_rep = rid_rep.unsqueeze(1).expand(-1, len(rid_rep), -1)    # [S(B_r), B_r, E]
        cid_rep = cid_rep.permute(1, 0, 2)    # [S, B_c, E]
        rest = self.fusion_encoder(rid_rep, cid_rep)    # [S(B), B_r, E]
        rest = rest.permute(1, 2, 0)    # [B_r, E, S(B)]

        cid_rep_ = cid_rep_.unsqueeze(1)    # [B_c, 1, E]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)    # [B, S(B)]
        return dp

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        # 
        cid_rep = self.ctx_encoder(cid, cid_mask, hidden=True)    # [1, S, E]
        cid_rep_ = cid_rep[:, 0, :]    # [1, E]
        rid_rep = self.can_encoder(rid, rid_mask)     # [10, E]
        rid_rep = rid_rep.unsqueeze(1)    # [10, 1, E]
        cid_rep = cid_rep.permute(1, 0, 2)    # [S, 1, E]
        rest = self.fusion_encoder(rid_rep, cid_rep)    # [10, 1, E]
        rest = rest.permute(1, 2, 0)    # [1, E, 10]

        cid_rep_ = cid_rep_.unsqueeze(1)    # [1, 1, E]
        dp = torch.bmm(cid_rep_, rest).squeeze()    # [10]
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dp = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc
