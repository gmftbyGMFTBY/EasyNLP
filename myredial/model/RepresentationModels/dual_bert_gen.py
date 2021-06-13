from model.utils import *


'''seq2seq + retrieval constrastive learning'''


class Seq2SeqModel(nn.Module):

    def __init__(self, model):
        super(Seq2SeqModel, self).__init__()
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model, model)
        # bert-ft
        self.model.encoder.resize_token_embeddings(self.model.encoder.config.vocab_size+1)
        self.model.decoder.resize_token_embeddings(self.model.decoder.config.vocab_size+1)

    def forward(self, cid, cid_mask, rid, rid_mask):
        outputs = self.model(
            input_ids=cid, 
            attention_mask=cid_mask, 
            decoder_input_ids=rid, 
            decoder_attention_mask=rid_mask
        )
        logits = outputs.logits    # [B, S, V]
        hidden = outputs.encoder_last_hidden_state[:, 0, :]    # [B, E]
        return logits, hidden


class BERTSeq2SeqDualEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTSeq2SeqDualEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']

        self.ctx_encoder = Seq2SeqModel(model)
        self.can_encoder = BertEmbedding(model=model)
        self.label_smooth_loss_fct = LabelSmoothLoss(smoothing=s)
        # bert-base-chinese, bert-base-uncased, hfl/chinese-roberta-wwm-ext use 0 for padding token
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        gen_logits, cid_rep = self.ctx_encoder(cid, cid_mask, rid, rid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return gen_logits, cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask, rid, rid_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask, rid, rid_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        _, cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        labels = batch['rids']    # [B, S]

        batch_size = cid.shape[0]
        gen_logits, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product

        # generative loss
        # gen_logits: [B, S, V]; label: [B, S]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        gen_loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # constrastive loss
        # mask = torch.zeros_like(dot_product).cuda()
        # mask[range(batch_size), range(batch_size)] = 1. 
        # loss = F.log_softmax(dot_product, dim=-1) * mask
        # loss = (-loss.sum(dim=1)).mean()

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        cl_loss = self.label_smooth_loss_fct(dot_product, gold)
        
        # total loss
        loss = cl_loss + gen_loss

        # cl acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.float)
        counter, sum_ = 0, 0
        for i, j in zip(shift_labels.view(-1), gen_acc):
            if i != 0:
                sum_ += 1
                counter += j.item()
        gen_acc = counter/sum_
        return (cl_loss, gen_loss, loss), (acc, gen_acc)
