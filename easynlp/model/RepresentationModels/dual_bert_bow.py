from model.utils import *

class BERTDualBOWEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualBOWEncoder, self).__init__()

        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.cls, self.sep, self.unk, self.pad = self.vocab.convert_tokens_to_ids(['[CLS]', '[SEP]', '[UNK]', '[PAD]'])
        self.token_embeddings = nn.Parameter(torch.randn((len(self.vocab), 768)))

        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = F.normalize(rid_rep, dim=-1)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep = F.normalize(cid_rep, dim=-1)
        return cid_rep
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product = (dot_product + 1)/2
        return dot_product

    def get_bow_loss(self, reps, ids):
        # phrase_reps: [B, H]; phrase_ids: [B, S]
        ## token mask to ignore the special tokens
        mask_unk = (ids != self.unk)
        mask_pad = (ids != self.pad)
        mask_sep = (ids != self.sep)
        mask_cls = (ids != self.cls)
        mask = mask_pad & mask_unk & mask_cls & mask_sep
        mask = mask.to(torch.long)

        logits = F.log_softmax(torch.matmul(reps, self.token_embeddings.t()), dim=-1)    # [B, V]
        target_logits = torch.gather(logits, 1, ids)    # [B, S]
        assert target_logits.size() == ids.size()
        target_logits = target_logits * mask    # [B, S]
        length = mask.sum(dim=-1)
        bow_loss = -(target_logits.sum(dim=-1) / length).mean()
        return bow_loss

    @torch.no_grad()
    def predict_acc(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dp = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        return dp

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        # cid_bow_loss = self.get_bow_loss(cid_rep, rid)
        rid_bow_loss = self.get_bow_loss(rid_rep, rid)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, rid_bow_loss, acc
