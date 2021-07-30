from model.utils import *

class BERTDualMixUpEncoder(nn.Module):

    '''Sentence-level MixUp data augmentation technique is used'''

    def __init__(self, **args):
        super(BERTDualMixUpEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)

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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    def mixup(self, x, y, alpha=1.0):
        '''https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py'''
        bsz = x.size()[0]
        lam = torch.zeros_like(x)    # [B, E]
        for i in range(bsz):
            for j in range(x.size()[1]):
                lam[i, j] = np.random.beta(alpha, alpha)
        index = torch.randperm(bsz)
        # x, y: [B, E]
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0] * 2
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        mixed_cid_rep, mixed_rid_rep = self.mixup(cid_rep, rid_rep)
        cid_rep = torch.cat([cid_rep, mixed_cid_rep], dim=0)
        rid_rep = torch.cat([rid_rep, mixed_rid_rep], dim=0)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
