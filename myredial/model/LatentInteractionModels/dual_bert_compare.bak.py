from model.utils import *

class BERTDualCompEncoder(nn.Module):

    '''This model needs the gray(hard negative) samples, which cannot be used for recall'''
    
    def __init__(self, **args):
        super(BERTDualCompEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        self.gray_num = args['gray_cand_num']
        nhead = args['nhead']
        dim_feedforward = args['dim_feedforward']
        dropout = args['dropout']
        num_encoder_layers = args['num_encoder_layers']

        # ====== Model ====== #
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        hidden_size = self.ctx_encoder.model.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size*2, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(2*hidden_size)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        self.trs_head = nn.Sequential(
            self.trs_encoder,
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2, hidden_size),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        b_c = len(cid_rep)
        rid_reps = rid_rep.unsqueeze(0).repeat(b_c, 1, 1)    # [B_c, B_r*gray, E]
        # fuse context into the response
        cid_reps = cid_rep.unsqueeze(1).repeat(1, len(rid), 1)    # [B_c, B_r*gray, E]
        for_comp = torch.cat([rid_reps, cid_reps], dim=-1)   # [B_c, B_r*gray, 2*E]
        comp_reps = self.trs_head(for_comp.permute(1, 0, 2)).permute(1, 0, 2)    # [B, G, E] 

        rid_reps = torch.cat([rid_reps, comp_reps], dim=-1)    # [B, G, 2*E]
        # rid_reps = rid_reps + comp_reps
        score = self.cls_head(rid_reps).squeeze(-1)    # [B, G]
        return score

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
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid)

        batch_size = rid.shape[0]
        scores = self._encode(cid, rid, cid_mask, rid_mask)
        scores = F.sigmoid(scores)
        return scores.squeeze(0)    # [G] = [B_r*gray]
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        b_c, b_r = len(cid), int(len(rid)//(self.gray_num+1))
        assert b_c == b_r
        scores = self._encode(cid, rid, cid_mask, rid_mask)    # [B_c, B_r*gray]
        mask = torch.zeros_like(scores)
        mask[torch.arange(b_c), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        loss = self.criterion(scores, mask)

        # acc
        acc_num = (scores.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / b_c

        return loss, acc
