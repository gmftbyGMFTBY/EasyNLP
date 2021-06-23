from model.utils import *

class SimCSE(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(SimCSE, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        self.ctx_encoder = BertEmbedding(model=model)
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)

    def _encode(self, cid, cid_mask):
        cid_rep_1 = self.ctx_encoder(cid, cid_mask)
        cid_rep_2 = self.ctx_encoder(cid, cid_mask)
        return cid_rep_1, cid_rep_2

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        return self.get_ctx(ids, attn_mask)

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        '''SimCSE doesn"t follow the test_model in the RepresentationModels'''
        pass
    
    def forward(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']

        batch_size = cid.shape[0]
        cid_rep_1, cid_rep_2 = self._encode(cid, cid_mask)
        dot_product = torch.matmul(cid_rep_1, cid_rep_2.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        loss = self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
