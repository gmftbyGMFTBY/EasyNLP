from model.utils import *

class BERTDualCLGatedEncoder(nn.Module):

    '''the modification of the dual-bert-cl model, which fusion the context information into the response representation to improve the recall performance:
        
    In this settings, the test set should be remeasured'''

    def __init__(self, **args):
        super(BERTDualCLGatedEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        p = args['dropout']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)

        # Gated module
        self.gate = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, cids, cid_attn_mask, ids, attn_mask):
        cid_rep = self.ctx_encoder(cids, cid_attn_mask)
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep_ = self.fusion_head(
            torch.cat([cid_rep, rid_rep], dim=-1)        
        )
        rid_rep += rid_rep_
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

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        # cid_cand_rep = self.ctx_encoder(cid_cand, cid_cand_mask)
        # rid_rep_ = self.fusion_head(
        #     torch.cat([cid_cand_rep, rid_rep], dim=-1)        
        # )
        # rid_rep += rid_rep_
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_cand = batch['ids_cand']

        cid_cand_mask = batch['ids_cand_mask']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_cand_rep = self.ctx_encoder(cid_cand, cid_cand_mask)    # [B, E]
        gate_score = self.gate(
            torch.cat([cid_cand_rep, rid_rep], dim=-1)        
        )     # [B, E]
        rid_rep = gate_score * cid_cand_rep + (1 - gate_score) * rid_rep

        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product
        dot_product_2 = torch.matmul(cid_cand_rep, rid_rep.t())     # [B, B]
        dot_product_2 /= np.sqrt(768)     # scale dot product
        # context constrastive loss
        ctx_dot_product = torch.matmul(cid_rep, cid_cand_rep.t())
        ctx_dot_product /= np.sqrt(768)

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        loss = self.label_smooth_loss(dot_product, gold)
        loss += self.label_smooth_loss(dot_product_2, gold)
        loss += self.label_smooth_loss(ctx_dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
