from model.utils import *

class XMoCoEncoder(nn.Module):

    '''https://aclanthology.org/2021.acl-long.477.pdf'''

    def __init__(self, **args):
        super(XMoCoEncoder, self).__init__()

        model = args['pretrained_model']

        self.momentum_ratio = args['momentum_ratio']
        self.ctx_fast_encoder = BertEmbedding(model=model, add_tokens=1)
        self.ctx_slow_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_fast_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_slow_encoder = BertEmbedding(model=model, add_tokens=1)

        # queue
        self.ctx_queue = nn.Parameter(
            torch.zeros(
                args['queue_size'], 768
            )
        )
        self.can_queue = nn.Parameter(
            torch.zeros(
                args['queue_size'], 768
            )
        )
        self.ctx_queue_idx, self.can_queue_idx = 0, 0
        self.ctx_queue_size, self.can_queue_size = 0, 0
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_fast_rep = self.ctx_fast_encoder(cid, cid_mask)
        rid_fast_rep = self.can_fast_encoder(rid, rid_mask)
        with torch.no_grad():
            # do not update the slow encoders
            cid_slow_rep = self.ctx_slow_encoder(cid, cid_mask)
            rid_slow_rep = self.can_slow_encoder(rid, rid_mask)
        return cid_fast_rep, cid_slow_rep, rid_fast_rep, rid_slow_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_fast_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_fast_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, _, rid_rep, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    @torch.no_grad()
    def push_queue(self, batch, ctx=True):
        '''return the last index of the pushed samples'''
        # batch: [B, E]
        bsz = len(batch)
        start = self.ctx_queue_idx if ctx else self.can_queue_idx
        if ctx:
            self.ctx_queue[start:start+bsz] = batch
            last_index = []
            for i in range(bsz):
                i = (self.ctx_queue_idx + i) % self.args['queue_size']
                last_index.append(i)
            self.ctx_queue_idx = (last_index[-1] + 1) % self.args['queue_size']
            self.ctx_queue_size = min(self.args['queue_size'], self.ctx_queue_size + bsz)
        else:
            self.can_queue[start:start+bsz] = batch
            last_index = []
            for i in range(bsz):
                i = (self.can_queue_idx + i) % self.args['queue_size']
                last_index.append(i)
            self.can_queue_idx = (last_index[-1] + 1) % self.args['queue_size']
            self.can_queue_size = min(self.args['queue_size'], self.can_queue_size + bsz)
        return last_index

    @torch.no_grad()
    def collect_negative_from_queue(self, ctx=True):
        if ctx:
            # collect from ctx_queue
            batch = self.ctx_queue[:self.ctx_queue_size]
        else:
            # collect from can_queue
            batch = self.can_queue[:self.can_queue_size]
        return batch

    def obtain_contrastive_loss(self, cid_rep, rid_rep, last_index):
        batch_size = len(cid_rep)
        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B2] 
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), last_index] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(last_index).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc

    @torch.no_grad()
    def momentum_update(self):
        # 1. context encoders update
        for param_q, param_k in zip(
            self.ctx_fast_encoder.parameters(), 
            self.ctx_slow_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum_ratio + param_q.data * (1. - self.momentum_ratio)
        # 2. response encoders update
        for param_q, param_k in zip(
            self.can_fast_encoder.parameters(), 
            self.can_slow_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum_ratio + param_q.data * (1. - self.momentum_ratio)
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, cid_s_rep, rid_rep, rid_s_rep = self._encode(cid, rid, cid_mask, rid_mask)

        # momentum update
        self.momentum_update()

        # distributed collect
        cid_s_rep, rid_s_rep = distributed_collect(cid_s_rep, rid_s_rep)
        # push the slow vectors into the queues
        last_cid_index = self.push_queue(cid_s_rep, ctx=True)
        last_rid_index = self.push_queue(rid_s_rep, ctx=False)

        # collect negative samples: [K, E]
        cid_neg_rep, rid_neg_rep = self.collect_negative_from_queue(ctx=True), self.collect_negative_from_queue(ctx=False)

        # contrastive loss:
        # 1. fast-q vs. slow-r
        loss_1, _ = self.obtain_contrastive_loss(cid_rep, rid_neg_rep, last_rid_index)
        # 2. fast-r vs. slow-q
        loss_2, _= self.obtain_contrastive_loss(rid_rep, cid_neg_rep, last_cid_index)
        # 3. fast-q vs. fast-r
        loss_3, acc = self.obtain_contrastive_loss(cid_rep, rid_rep, range(len(cid)))
        # sum the loss
        loss = loss_1 + loss_2 + loss_3
        return loss, acc
