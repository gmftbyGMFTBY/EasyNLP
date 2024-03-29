from model.utils import *

    
class BERTDualGrayFullEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualGrayFullEncoder, self).__init__()
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num']

        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

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
        cid, cid_mask = batch['ids'], batch['ids_mask']
        rid, rid_mask = batch['rids'], batch['rids_mask'] 

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)

        # during deploy, add the softmax for normalization
        dot_product /= np.sqrt(768)
        dot_product = (dot_product - dot_product.min()) / (1e-3 + dot_product.max() - dot_product.min())
        return dot_product
    
    def forward(self, batch):
        cid, cid_mask = batch['ids'], batch['ids_mask']
        rid, rid_mask = batch['rids'], batch['rids_mask']
        ipdb.set_trace()

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, 10*B]

        # loss
        mask = torch.zeros_like(dot_product).cuda()
        mask[torch.arange(batch_size), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).to(torch.float).mean().item()
        return loss, acc
