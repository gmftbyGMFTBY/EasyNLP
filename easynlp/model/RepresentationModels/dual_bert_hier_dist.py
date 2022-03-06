from model.utils import *
from .dual_bert import *
from .dual_bert_hier import *


class DRBERTV2Encoder(nn.Module):

    def __init__(self, **args):
        super(DRBERTV2Encoder, self).__init__()
        # teacher model
        self.dr_bert = BERTDualEncoder(**args) 
        # student model
        self.dr_bert_v2 = BERTDualHierarchicalTrsEncoder(**args)
        self.args = args
        self.gray_cand_num = 1
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def _encode(self, 
        cid, rid, cid_mask, rid_mask,
        cid_, rid_, cid_mask_, rid_mask_, turn_length
    ):
        # teacher distribution
        with torch.no_grad():
            cid_rep, rid_rep = self.dr_bert._encode(cid, rid, cid_mask, rid_mask)
            cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
            dp = torch.matmul(cid_rep, rid_rep.t())    # [B_c, B_r]

        # student distribution
        cid_reps, rid_rep = self.dr_bert_v2._encode(cid_, rid_, cid_mask_, rid_mask_, turn_length)
        cid_rep = self.dr_bert_v2.get_context_level_rep(cid_reps, turn_length)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dp_ = torch.matmul(cid_rep, rid_rep.t())    # [B_c, B_r]
        return dp, dp_
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        # dr-bert-v2 model
        batch_size = rid.shape[0]
        cid_reps, rid_rep = self.dr_bert_v2._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.dr_bert_v2.get_context_level_rep(cid_reps, turn_length)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        # dr-bert
        cid, rid, cid_mask, rid_mask = batch['ids'], batch['rids'], batch['ids_mask'], batch['rids_mask']
        # dr-bert-v2
        cid_, rid_, cid_mask_, rid_mask_ = batch['ids_'], batch['rids_'], batch['ids_mask_'], batch['rids_mask_']
        turn_length = batch['turn_length']

        dp, dp_ = self._encode(
            cid, rid, cid_mask, rid_mask, 
            cid_, rid_, cid_mask_, rid_mask_, turn_length
        )

        # distillation learning
        kl_loss = self.criterion(F.log_softmax(dp_, dim=-1), F.softmax(dp, dim=-1))

        # supervised loss
        ## mask the hard negative
        bsz, rbsz = dp_.size()
        mask = torch.zeros_like(dp_)
        mask[range(bsz), range(rbsz)] = 1.
        loss_ = F.log_softmax(dp_, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # overall loss
        loss += kl_loss

        # acc
        acc = (dp_.max(dim=-1)[1] == torch.LongTensor(torch.arange(rbsz)).cuda()).to(torch.float).mean().item()
        return loss, acc

