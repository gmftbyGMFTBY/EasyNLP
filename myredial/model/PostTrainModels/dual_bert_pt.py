from model.utils import *
from dataloader.util_func import *

class BERTDualPTEncoder(nn.Module):

    '''constrastive learning loss and MLM loss'''
    
    def __init__(self, **args):
        super(BERTDualPTEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertForMaskedLM.from_pretrained(model)
        self.ctx_encoder.resize_token_embeddings(self.ctx_encoder.config.vocab_size+1)
        self.can_encoder = BertForMaskedLM.from_pretrained(model)
        self.can_encoder.resize_token_embeddings(self.can_encoder.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.ctx_encoder.config.vocab_size

    def _encode(self, cid, rid, cid_mask, rid_mask):
        output = self.ctx_encoder(
            input_ids=cid,
            attention_mask=cid_mask,
            output_hidden_states=True,
        )
        cid_pred = output.logits
        cid_rep = output.hidden_states[-1][:, 0, :]
        output = self.can_encoder(
            input_ids=rid,
            attention_mask=rid_mask,
            output_hidden_states=True,
        )
        rid_pred = output.logits
        rid_rep = output.hidden_states[-1][:, 0, :]
        # *id_prediction_scores: [B, S, V]
        # *id_rep: [B, E]
        return cid_pred, rid_pred, cid_rep, rid_rep

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
        cid_pred, rid_pred, cid_rep, rid_rep = self._encode(
            cid, rid, cid_mask, rid_mask
        )

        # mlm loss
        cids_mlm_loss = self.criterion(
            cid_pred.view(-1, self.vocab_size),
            cid_mask_label.view(-1)
        )
        rids_mlm_loss = self.criterion(
            rid_pred.view(-1, self.vocab_size),
            rid_mask_label.view(-1)
        )
        mlm_loss = cids_mlm_loss + rids_mlm_loss

        # mlm acc
        token_acc_num, total_num = 0, 0
        a, b = self.calculate_token_acc(cid_mask_label, cid_pred)
        c, d = self.calculate_token_acc(rid_mask_label, rid_pred)
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
