from model.utils import *


class BERTDualCompEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualCompEncoder, self).__init__()
        model = args['pretrained_model']
        nhead = args['nhead']
        dim_feedforward = args['dim_ffd']
        dropout = args['dropout']
        num_encoder_layers = args['num_encoder_layers']
        self.gray_num = args['gray_cand_num']

        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            2*768, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(2*768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )

        self.trs_head = nn.Sequential(
            self.trs_encoder,
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(768*2, 768)
        )

        self.gate_head = nn.Sequential(
            nn.Linear(768*3, 768*2),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(768*2, 768),
            nn.Sigmoid(),
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [b_c, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [b_r, E]

        b_c, b_r = len(cid_rep), len(rid_rep)
        rid_rep_ = rid_rep.unsqueeze(0).repeat(b_c, 1, 1)    # [b_c, b_r, e]
        cid_rep_ = cid_rep.unsqueeze(1).repeat(1, b_r, 1)    # [b_c, b_r, e]
        cross_rep = torch.cat([cid_rep_, rid_rep_], dim=-1)    # [b_c, b_r, 2*e]
        cross_rep = self.trs_head(cross_rep.permute(1, 0, 2)).permute(1, 0, 2)    # [b_r, b_c, 2*e] -> [b_c, b_r, e]
        gate = self.gate_head(
            torch.cat([
                rid_rep_,    # [b_c, b_r, e]
                cid_rep_,    # [b_c, b_r, e]
                cross_rep,    # [b_c, b_r, e]
            ])
        )    # [b_c, b_r, e]

        final_rid_rep = gate * rid_rep_ + (1 - gate) * cross_rep    # [b_c, b_r, e]
        final_rid_rep = final_rid_rep.permute(0, 2, 1)    # [b_c, e, b_r]
        cid_rep_ = cid_rep.unsqueeze(1)    # [b_c, 1, e]
        # [b_c, 1, e] x [b_c, e, b_r]
        dot_product = torch.bmm(cid_rep, final_rid_rep).squeeze(1)    # [b_c, b_r]
        return dot_product

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask).squeeze(dim=0)    # [b_r]
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        # [b_c, b_r]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[torch.arange(batch_size), torch.arange(0, len(rid), self.gray_num+1)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        self.ctx_encoder.load_bert_model(state_dict)
        self.can_encoder.load_bert_model(state_dict)
        print(f'[!] load checkpoint for candidate and context encoder')
