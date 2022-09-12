from model.utils import *

class TransformerDualEncoder(nn.Module):

    def __init__(self, **args):
        super(TransformerDualEncoder, self).__init__()
        model = args['pretrained_model']
        config = BertConfig.from_pretrained(model)
        config.n_layer = args['n_layer']
        self.ctx_encoder = BertModel.from_pretrained(model, num_hidden_layers=args['n_layer'])
        self.can_encoder = BertModel.from_pretrained(model, num_hidden_layers=args['n_layer'])
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(input_ids=cid, attention_mask=cid_mask).last_hidden_state[:, 0, :]
        rid_rep = self.can_encoder(input_ids=rid, attention_mask=rid_mask).last_hidden_state[:, 0, :]
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(input_ids=ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(input_ids=ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
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

    @torch.no_grad()
    def predict_acc(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dp = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        return dp

    @torch.no_grad()
    def score(self, batch):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        cid, rid = batch['ids_'], batch['rids_']
        cid_mask, rid_mask = batch['ids_mask_'], batch['rids_mask_']
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        score = torch.einsum('ij,ij->i', cid_rep, rid_rep)
        return score
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

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

        return loss, acc
