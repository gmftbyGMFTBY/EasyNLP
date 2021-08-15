from model.utils import *
from dataloader.util_func import *

class BERTDualPTEncoder(nn.Module):

    '''constrastive learning loss and MLM loss'''
    
    def __init__(self, **args):
        super(BERTDualPTEncoder, self).__init__()
        model = args['pretrained_model']
        # return [B, S, E]
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.ctx_encoder.model.config.vocab_size
        self.vocab = BertTokenizer.from_pretrained(model)

    def _encode(self, cid, rid, cid_mask_ids, rid_mask_ids, cid_mask, rid_mask):
        cid_reps = self.ctx_encoder(cid_mask_ids, cid_mask)    # [B, S, E]
        rid_reps = self.can_encoder(rid_mask_ids, rid_mask)    # [B, S, E]
        cid_rep, rid_rep = cid_reps[:, 0, :], rid_reps[:, 0, :]
        return cid_rep, rid_rep, cid_reps, rid_reps

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
        cid_mask_label = batch['mask_labels_ids']    # [B, S]
        rid_mask_label = batch['mask_labels_rids']
        batch_size = len(cid)
        ipdb.set_trace()
        cid_rep, rid_rep, cid_reps, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)

        # mlm loss
        cids_mlm_loss = self.criterion(
            cid_reps.view(-1, self.vocab_size),
            cid_mask_label.view(-1)
        )
        rids_mlm_loss = self.criterion(
            rid_reps.view(-1, self.vocab_size),
            rid_mask_label.view(-1)
        )
        mlm_loss = cids_mlm_loss + rids_mlm_loss

        # mlm acc
        token_acc_num, total_num = 0, 0
        a, b = self.calculate_token_acc(cid_mask_label, cid_lm)
        c, d = self.calculate_token_acc(rid_mask_label, rid_lm)
        token_acc = (a+c)/(b+d)

        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        cl_loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num/batch_size

        return mlm_loss, cl_loss, token_acc, acc
