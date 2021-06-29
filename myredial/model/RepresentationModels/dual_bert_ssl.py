from model.utils import *

class BERTDualSSLEncoder(nn.Module):

    '''Dual bert with self supervised learning:
        1. load the pre-trained dual-bert model
        2. start the ssl training
    '''
    
    def __init__(self, **args):
        super(BERTDualSSLEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        self.gray_num = args['gray_cand_num']
        self.radius = args['radius']
        self.low_conf = args['low_conf']

        # ssl counter
        # self.ssl_interval_step has been set
        self.current_step = 0

        # ====== Double Network ===== #
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder_shadow = BertEmbedding(model=model)
        self.can_encoder_shadow = BertEmbedding(model=model)

    def update(self):
        if self.current_step == self.ssl_interval_step:
            self.current_step = 0
            self.copy()    # copy the parameter
        self.current_step += 1

    def copy(self):
        # copy the parameters from original models to the shadow models
        self.ctx_encoder_shadow.load_state_dict(
            self.ctx_encoder.state_dict()
        )
        self.can_encoder_shadow.load_state_dict(
            self.can_encoder.state_dict()
        )
        print(f'[!] update the shadow models parameters over')

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder_shadow(cid, cid_mask)
        rid_rep = self.can_encoder_shadow(rid, rid_mask)
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

    @torch.no_grad()
    def set_pseudo_label(self, cid, rid, cid_mask, rid_mask):
        '''ids: [B, E]; rids: [B*(gray+1), E];'''
        cid_rep, rid_rep = self._encode_(cid, rid, cid_mask, rid_mask)
        # cosine similarity
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        matrix = torch.matmul(cid_rep, rid_rep.t())    # [B, B*(gray+1)]

        eye = matrix[torch.arange(len(cid)), torch.arange(0, len(rid), self.gray_num+1)].unsqueeze(1)
        mask = (matrix > eye - self.radius)
        label = torch.where(mask, torch.ones_like(matrix), torch.zeros_like(matrix))

        # give the non-diag item the lower weight
        weight_matrix = torch.ones_like(label) * self.low_conf
        weight_matrix[torch.arange(len(cid)), torch.arange(0, len(rid), self.gray_num+1)] = 1.

        label *= weight_matrix
        return label
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]

        # use label smoothing to avoid the over confidence
        mask = self.set_pseudo_label(cid, rid, cid_mask, rid_mask)
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=-1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
