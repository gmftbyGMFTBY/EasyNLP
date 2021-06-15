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

        # bow loss (ignore [PAD] and duplicate tokens)
        cid_bow = F.log_softmax(self.bow_head(cid_rep, dim=-1))
        rid_bow = F.log_softmax(self.bow_head(rid_rep, dim=-1))
        bow_loss = 0
        for cid_bow_, cid_, rid_bow_, rid_ in zip(cid_bow, cid, rid_bow, rid):
            cid_ = [i for i in list(set(cid_.tolist())) if i not in [0]]
            rid_ = [i for i in list(set(rid_.tolist())) if i not in [0]]
            cid_, rid_ = torch.LongTensor(cid_), torch.LongTensor(rid_)
            if torch.cuda.is_available():
                cid_, rid_ = cid_.cuda(), rid_.cuda()
            loss = torch.index_select(cid_bow_, 0, cid_).mean() + torch.index_select(rid_bow_, 0, rid_).mean()
            bow_loss -= loss
        loss = bow_loss / len(cid)

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        loss += self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
