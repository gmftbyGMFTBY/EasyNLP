from model.utils import *

class BERTDualSEEDEncoder(nn.Module):

    '''refer to emnlp 2021 paper for more details: less is more: pre-train a strong text encoder for dense retrieval using a weak decoder'''

    def __init__(self, **args):
        super(BERTDualSEEDEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertSEEDEmbedding(
            model=model, 
            add_tokens=1, 
            dropout=args['dropout'],
            vocab_size=args['vocab_size'],
            nhead=args['nhead'],
            nlayer=args['nlayer'],
            attn_span=args['attention_span']
        )
        self.can_encoder = BertSEEDEmbedding(
            model=model, 
            add_tokens=1,
            dropout=args['dropout'],
            vocab_size=args['vocab_size'],
            nhead=args['nhead'],
            nlayer=args['nlayer'],
            attn_span=args['attention_span']
        )
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep, cid_de_loss, cid_de_acc = self.ctx_encoder(cid, cid_mask)
        rid_rep, rid_de_loss, rid_de_acc = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep, cid_de_loss, cid_de_acc, rid_de_loss, rid_de_acc

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep, _, _ = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep, _, _ = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep, _, _, _, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep, cid_de_loss, cid_de_acc, rid_de_loss, rid_de_acc = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        loss += cid_de_loss + rid_de_loss

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc, cid_de_acc, rid_de_acc
