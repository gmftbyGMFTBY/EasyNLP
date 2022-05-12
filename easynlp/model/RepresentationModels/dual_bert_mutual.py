from model.utils import *

class BERTDualMutualDatasetEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualMutualDatasetEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=3)
        self.can_encoder = BertEmbedding(model=model, add_tokens=3)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.pad_token_id
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
        cid = batch['ids']
        rid = batch['rids']
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid)
        rid_mask = generate_mask(rid)
        cid, cid_mask, rid, rid_mask = to_cuda(cid, cid_mask, rid, rid_mask)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rids = batch['rids']
        hn_rids = batch['hn_rids']

        cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
        hn_rids = list(chain(*hn_rids))
        rids = rids + hn_rids
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        cid_mask = generate_mask(cid)
        rid_mask = generate_mask(rids)
        cid, rids, cid_mask, rid_mask = to_cuda(cid, rids, cid_mask, rid_mask)
        cid_rep, rid_rep = self._encode(cid, rids, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        batch_size = len(cid)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
