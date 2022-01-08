from model.utils import *

class SimCSESEEDEncoder(nn.Module):

    '''use the weight of the bert-fp, and only the masked lm loss is used'''

    def __init__(self, **args):
        super(SimCSESEEDEncoder, self).__init__()
        self.encoder = simcse_model(
            pretrained_model=args['pretrained_model'], 
            dropout=args['dropout']
        )
        self.decoder = WeakTrsDecoder(
            args['dropout'],
            args['vocab_size'],
            args['nhead'],
            args['nlayer'],
            args['attention_span'],
        )
        self.args = args

    def forward(self, batch):
        inpt = batch['ids']
        tids = batch['tids']
        attn_mask = batch['ids_mask']

        # semantic contrastive learning
        rep1 = self.encoder(inpt, attn_mask, tids)
        rep2 = self.encoder(inpt, attn_mask, tids)
        # rep1 = hidden1[:, 0, :] / hidden1[:, 0, :].norm(dim=-1, keepdim=True)
        # rep2 = hidden2[:, 0, :] / hidden2[:, 0, :].norm(dim=-1, keepdim=True)
        dp = torch.matmul(rep1, rep2.t())

        mask = torch.zeros_like(dp)
        mask[range(len(dp)), range(len(dp))] = 1.
        loss = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss.sum(dim=-1)).mean()

        cl_acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(len(dp))).cuda()).to(torch.float).mean().item()

        # weak decoder loss and acc
        de_loss_1, de_acc_1 = self.decoder(
            inpt,
            hidden1.permute(1, 0, 2)[:1, :, :]    # [S, B, E], only the CLS token embedding is used
        )
        de_loss_2, de_acc_2 = self.decoder(
            inpt,
            hidden2.permute(1, 0, 2)[:1, :, :]    # [S, B, E]
        )
        de_loss = de_loss_1 + de_loss_2
        de_acc = (de_acc_1 + de_acc_2) / 2

        return loss, de_loss, cl_acc, de_acc
    
    @torch.no_grad()
    def get_embedding(self, ids, tids, ids_mask):
        self.encoder.eval()
        rep = self.encoder(ids, ids_mask, tids)[:, 0, :]    # [B, E]
        rep /= rep.norm(dim=-1, keepdim=True)
        return rep

    @torch.no_grad()
    def predict(self, ids, tids, ids_mask, ids_1, tids_1, ids_mask_1):
        self.encoder.eval()
        bsz, _ = ids.size()
        s1 = self.get_embedding(ids, tids, ids_mask)
        s2 = self.get_embedding(ids_1, tids_1, ids_mask_1) 
        scores = torch.matmul(s1, s2.t())[range(bsz), range(bsz)]
        return scores.tolist()
