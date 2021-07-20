from model.utils import *
from dataloader.util_func import *

class BERTDualPTEncoder(nn.Module):

    '''constrastive learning loss and MLM loss'''
    
    def __init__(self, **args):
        super(BERTDualPTEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertForPreTraining.from_pretrained(model=model)
        self.can_encoder = BertForPreTraining.from_pretrained(model=model)
        # [EOS] token
        self.ctx_encoder.resize_token_embeddings(self.ctx_encoder.config.vocab_size+1)
        self.can_encoder.resize_token_embeddings(self.can_encoder.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.ctx_encoder.model.config.vocab_size

    def _encode(self, cid, rid, cid_mask_ids, rid_mask_ids, cid_mask, rid_mask):
        cid_output = self.ctx_encoder(cid_mask_ids, cid_mask)
        rid_output = self.can_encoder(rid_mask_ids, rid_mask)

        cid_mlm_logits, cid_cls_logits = cid_output.prediction_logits, cid_output.seq_relationship_logits
        rid_mlm_logits, rid_cls_logits = rid_output.prediction_logits, rid_output.seq_relationship_logits
        return cid_rep[:, 0, :], rid_rep[:, 0, :], cid_lm, rid_lm

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
        cid_rep, rid_rep, _, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    def calculate_token_acc(self, mask_label, logits):
        not_ignore = mask_label.ne(-1)
        num_targets = not_ignore.sum().item()
        correct = (F.softmax(logits, dim=-1).max(dim=-1)[1] == mask_label) & not_ignore
        correct = correct.sum().item()
        return correct, num_targets
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        cid_mask_label = batch['ids_mask_label']    # [B, S]
        rid_mask_label = batch['rids_mask_label']

        inner_bsz = int(len(cid)/2)
        cid_rep, rid_rep, cid_lm, rid_lm = self._encode(cid, rid, cid_mask, rid_mask)

        # mlm loss
        cids_mlm_loss = self.criterion(
            cid_lm.view(-1, self.vocab_size),
            cid_mask_label.view(-1)
        )
        rids_mlm_loss = self.criterion(
            rid_lm.view(-1, self.vocab_size),
            rid_mask_label.view(-1)
        )
        mlm_loss = cids_mlm_loss + rids_mlm_loss

        # mlm acc
        token_acc_num, total_num = 0, 0
        a, b = self.calculate_token_acc(cid_mask_label, cid_lm)
        c, d = self.calculate_token_acc(rid_mask_label, rid_lm)
        token_acc = (a+c)/(b+d)

        # constrastive loss
        cid_rep_1, cid_rep_2 = torch.split(cid_rep, inner_bsz)
        rid_rep_1, rid_rep_2 = torch.split(rid_rep, inner_bsz)
        assert len(cid_rep_1) == len(cid_rep_2)
        assert len(rid_rep_1) == len(rid_rep_2)

        # c-r
        dot_product1 = torch.matmul(cid_rep_1, rid_rep_1.t())     # [B, B]
        dot_product2 = torch.matmul(cid_rep_1, rid_rep_2.t())     # [B, B]
        dot_product3 = torch.matmul(cid_rep_2, rid_rep_1.t())     # [B, B]
        dot_product4 = torch.matmul(cid_rep_2, rid_rep_2.t())     # [B, B]
        # c-c
        dot_product5 = torch.matmul(cid_rep_1, cid_rep_2.t())     # [B, B]
        # r-r
        dot_product6 = torch.matmul(rid_rep_1, rid_rep_2.t())     # [B, B]

        # constrastive loss
        mask = torch.zeros_like(dot_product1)
        mask[range(inner_bsz), range(inner_bsz)] = 1. 
        cl_loss = 0
        loss_ = F.log_softmax(dot_product1, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()
        loss_ = F.log_softmax(dot_product2, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()
        loss_ = F.log_softmax(dot_product3, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()
        loss_ = F.log_softmax(dot_product4, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()
        loss_ = F.log_softmax(dot_product5, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()
        loss_ = F.log_softmax(dot_product6, dim=-1) * mask
        cl_loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product1, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(inner_bsz)).cuda()).sum().item()
        acc = acc_num/inner_bsz

        return mlm_loss, cl_loss, token_acc, acc
