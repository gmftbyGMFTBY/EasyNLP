from model.utils import *

class BERTDualBOWEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualBOWEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        p = args['dropout']
        vocab_size = args['vocab_size']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.bow_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=p),
            nn.Linear(768, vocab_size),
        )
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)

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
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product

        # bow loss
        cid_bow = F.log_softmax(self.bow_head(cid_rep))
        rid_bow = F.log_softmax(self.bow_head(rid_rep))
        ## sum (not include the [PAD])
        

        # constrastive loss
        # mask = torch.zeros_like(dot_product).cuda()
        # mask[range(batch_size), range(batch_size)] = 1. 
        # loss = F.log_softmax(dot_product, dim=-1) * mask
        # loss = (-loss.sum(dim=1)).mean()

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        loss = self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
