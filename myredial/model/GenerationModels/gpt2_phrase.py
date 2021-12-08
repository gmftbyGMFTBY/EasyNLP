from model.utils import *

class GPT2PhraseEncoder(nn.Module):

    def __init__(self, **args):
        super(GPT2PhraseEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls])

        self.gpt2_encoder = GPT2LMIRModel(model=gpt2_model)
        self.bert_encoder = BertFullEmbedding(model=model)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.test_max_len = args['test_max_len']
        self.temp = args['temp']

    def _encode(self, ids, ids_mask, bert_ids, bert_ids_mask):
        gpt2_logits, gpt2_rep = self.gpt2_encoder(ids, ids_mask)
        bert_rep = self.bert_encoder(bert_ids, bert_ids_mask)
        bert_rep = bert_rep[:, 1:-1, :]    # ignore the [CLS] and [SEP] tokens' embeddings
        return gpt2_logits, gpt2_rep, bert_rep

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.gpt2_encoder.model(
            input_ids=ids, 
            attention_mask=ids_mask
        )
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.gpt2_encoder.model(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        bert_ids, bert_ids_mask = batch['bert_ids'], batch['bert_ids_mask']
        batch_size, length = gpt2_ids.size()

        gpt2_logits, gpt2_rep, bert_rep = self._encode(gpt2_ids, gpt2_ids_mask, bert_ids, bert_ids_mask)
        ## gpt2 lm loss
        shift_logits = gpt2_logits[..., :-1, :].contiguous()
        shift_labels = gpt2_ids[..., 1:].contiguous()
        lm_loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        ## gpt2 token acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        token_acc = correct / num_targets

        ## token-aware contrastive training loss
        # bert_rep: [B, S-1, E]; gpt2_rep: [B, S-1, E]; gpt2_ids_mask: [B, S-1]
        length -= 1
        gpt2_ids_mask = gpt2_ids_mask[:, 1:]
        bert_rep = F.normalize(bert_rep[:, :-1, :], dim=-1)
        gpt2_rep = F.normalize(gpt2_rep[:, 1:, :],  dim=-1)
        cosine_sim = torch.bmm(gpt2_rep, bert_rep.permute(0, 2, 1))    # [B, S, S]
        cosine_sim /= self.temp

        # build the tacl loss
        mask = torch.zeros_like(cosine_sim)    # [B, S, S]
        mask[:, range(length), range(length)] = 1. 
        effective_num = 0
        # [PAD] must be ignored
        for i in range(batch_size):
            num_nonzero = gpt2_ids_mask[i].nonzero().size(0)
            mask[i][range(num_nonzero, length), range(num_nonzero, length)] = 0.
            effective_num += num_nonzero
        loss_ = F.log_softmax(cosine_sim, dim=-1) * mask    # [B, S, S]
        tacl_loss = -loss_.sum(dim=2)    # [B, S]
        tacl_loss = tacl_loss.view(-1).sum()    # [B*S]
        tacl_loss /= effective_num
        
        # tacl acc
        acc_num = 0
        for dp, mask_ in zip(cosine_sim, mask):
            # dp: [S, S]; mask_: [S, S]
            num_nonzero = mask_.nonzero().size(0)
            dp = dp[:num_nonzero, :num_nonzero]
            acc_num += (dp.max(dim=-1)[1] == torch.arange(num_nonzero).cuda()).sum().item()
        tacl_acc = acc_num / effective_num
        return lm_loss, tacl_loss, token_acc, tacl_acc
