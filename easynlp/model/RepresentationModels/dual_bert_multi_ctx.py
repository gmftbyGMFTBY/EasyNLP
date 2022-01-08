from model.utils import *

class BERTDualMultiContextEncoder(nn.Module):

    '''useless piece of shit!!!'''

    def __init__(self, **args):
        super(BERTDualMultiContextEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        # for debug
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')

    def _encode(self, cid, rid, cid_mask, rid_mask, cid_turn_length):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [M, E]
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, cid_turn_length)
        cid_rep = []
        dp = []
        for item in cid_reps:
            # item: [M, E]; rid_rep: [B, E]
            scores = torch.matmul(item, rid_rep.t())    # [M, B]
            scores = scores.mean(dim=0)    # [B]
            dp.append(scores)
        dp = torch.stack(dp)     # [B_c, B_r]
        return dp

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
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask, batch['ids_length'])
        dot_product = dot_product.squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(rid)

        dot_product = self._encode(cid, rid, cid_mask, rid_mask, batch['ids_length'])

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
