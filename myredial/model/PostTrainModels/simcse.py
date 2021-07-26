from model.utils import *

class SimCSE(nn.Module):

    '''two bert encoder are not shared, which is different from the original SimCSE model'''

    def __init__(self, **args):
        super(SimCSE, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']

        cid_rep, rid_rep = self._encode(ids, ids, ids_mask, ids_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        # distributed samples collected
        cid_reps, rid_reps = distributed_collect(cid_rep, rid_rep)
        dot_product = torch.matmul(cid_reps, rid_reps.t())     # [B, B]
        batch_size = len(cid_reps)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
