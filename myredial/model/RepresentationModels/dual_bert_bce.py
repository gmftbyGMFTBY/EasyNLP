from model.utils import *


class BERTDualEncoderBCE(nn.Module):

    '''Dual bert with hard negative samples, but the bce loss is used instead of the contrastive loss'''

    def __init__(self, **args):
        super(BERTDualEncoderBCE, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Linear(768, 1)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        bsz_c, bsz_r = len(cid_rep), len(rid_rep)
        cid_rep = cid_rep.unsqueeze(1).expand(-1, bsz_r, -1)    # [B_c, B_r, E]
        rid_rep = rid_rep.unsqueeze(0).expand(bsz_c, -1, -1)    # [B_c, B_r, E]
        rep = self.fusion_head(torch.cat([cid_rep, rid_rep], dim=-1)).squeeze(-1)    # [B_c, B_r]
        return rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask).squeeze(0)    # [B_r]
        dot_product = torch.sigmoid(dot_product)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        batch_size = len(cid)

        # BCE loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        dot_product, mask = dot_product.view(-1), mask.view(-1)
        a = dot_product.view(-1)
        b = mask.view(-1)

        # random shuffle
        random_idx = list(range(len(a)))
        random.shuffle(random_idx)
        a = torch.stack([a[i] for i in random_idx])
        b = torch.stack([b[i] for i in random_idx])
        loss = self.criterion(a, b)

        # acc
        acc = ((torch.sigmoid(dot_product).view(-1) > 0.5).float() == mask.view(-1)).float().mean().item()

        return loss, acc
