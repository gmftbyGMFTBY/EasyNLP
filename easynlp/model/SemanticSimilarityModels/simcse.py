from model.utils import *

class SimCSE(nn.Module):

    def __init__(self, **args):
        super(SimCSE, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, ids, ids_mask):
        rep_1 = self.encoder(ids, ids_mask)
        rep_2 = self.encoder(ids, ids_mask)
        rep_1, rep_2 = F.normalize(rep_1), F.normalize(rep_2)
        return rep_1, rep_2

    def compact_binary_vectors(self, ids):
        # ids: [B, D]
        ids = ids.cpu().numpy().astype('int')
        ids = np.split(ids, int(ids.shape[-1]/8), axis=-1)
        ids = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in ids]    
            ).transpose().astype('uint8')
        )
        return ids

    @torch.no_grad()
    def get_embedding(self, ids, ids_mask):
        rep = self.encoder(ids, ids_mask)
        rep = F.normalize(rep, dim=-1)
        return rep

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']

        rep_1, rep_2 = self._encode(ids, ids_mask)
        # distributed samples collected
        rep_1s, rep_2s = distributed_collect(rep_1, rep_2)
        dot_product = torch.matmul(rep_1s, rep_2s.t())     # [B, B]
        dot_product /= self.temp
        batch_size = len(rep_1s)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc

class WZSimCSE(nn.Module):

    '''two bert encoder are not shared, which is different from the original SimCSE model'''

    def __init__(self, **args):
        super(WZSimCSE, self).__init__()
        self.temp = args['temp']
        model = args['pretrained_model']
        self.encoder = simcse_model(pretrained_model=model, dropout=args['dropout'])
        self.supervised = args['is_supervised']
        self.args = args

    def _encode(self, ids, tids, ids_mask):
        rep_1 = self.encoder(ids, ids_mask, tids)[:, 0, :]
        rep_2 = self.encoder(ids, ids_mask, tids)[:, 0, :]
        rep_1, rep_2 = F.normalize(rep_1, dim=-1), F.normalize(rep_2, dim=-1)
        return rep_1, rep_2

    @torch.no_grad()
    def get_embedding(self, ids, tids, ids_mask):
        # return F.normalize(self.encoder(ids, ids_mask, tids)[:, 0, :], dim=-1)
        return self.encoder(ids, ids_mask, tids)[:, 0, :]

    @torch.no_grad()
    def predict(self, ids, tids, ids_mask, ids_2, tids_2, ids_mask_2):
        self.encoder.eval()
        bsz, _ = ids.size()
        s1 = self.get_embedding(ids, tids, ids_mask)    # [B, 768]
        s2 = self.get_embedding(ids_2, tids_2, ids_mask_2)    # [B, 768]
        scores = torch.matmul(s1, s2.t())[range(bsz), range(bsz)]    # [B]
        return scores.tolist()

    def forward(self, batch):
        if self.supervised:
            ids = batch['ids']
            tids = batch['tids']
            ids_mask = batch['ids_mask']
            ids_2 = batch['ids_2']
            tids_2 = batch['tids_2']
            ids_mask_2 = batch['ids_mask_2']
            rep_1 = self.encoder(ids, tids, ids_mask)[:, 0, :]
            rep_2 = self.encoder(ids_2, tids_2, ids_mask_2)[:, 0, :]
            dot_product = torch.matmul(rep_1, rep_2.t())     # [B, B]
            batch_size = len(rep_1)
        else:
            ids = batch['ids']
            tids = batch['tids']
            ids_mask = batch['ids_mask']
            rep_1, rep_2 = self._encode(ids, tids, ids_mask)
            dot_product = torch.matmul(rep_1, rep_2.t())     # [B, B]
            batch_size = len(rep_1)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
