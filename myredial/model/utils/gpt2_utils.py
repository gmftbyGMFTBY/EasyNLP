from .header import *

class GPT2CLHeadModel(nn.Module):

    def __init__(self, model_name, vocab_size, unk=None, pad=None, temp=0.07):
        super(GPT2CLHeadModel, self).__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.lm = nn.Parameter(torch.randn(768, vocab_size))
        self.vocab_size = vocab_size
        self.unk = unk
        self.pad = pad
        self.temp = temp

    def forward(self, ids, ids_mask):
        '''ids/ids_mask: [B, S]; label: [B, S]'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
        )
        rep = F.normalize(output.last_hidden_state, dim=-1)    # [B, S, E]
        rep = rep[:, :-1, :].contiguous()    # [B, S-1, E]
        ids_mask = ids_mask[:, :-1]    # [B, S-1]
        label = ids[:, 1:]    # [B, S-1]

        # contrastive loss
        dp = torch.matmul(rep, self.lm).view(-1, self.vocab_size)    # [B*S, V]
        dp /= self.temp
        mask = torch.zeros_like(dp)
        mask[range(len(dp)), label.reshape(-1)] = 1.    # [B*S, V]
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = -loss_.sum(dim=1)     # [B*S]
        # only non-padding tokens will be used for training
        loss = loss[~(ids_mask.reshape(-1) == self.pad)].mean()
        
        # acc (ignore padding tokens)
        acc = dp.max(dim=-1)[1] == label.reshape(-1)
        acc = acc[~(ids_mask.reshape(-1) == self.pad)].to(torch.float).mean().item()
        return loss, acc

    def predict_one_step(self, ids, ids_mask, pos_ids, past_key_values):
        '''use cache for speedup'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        rep = F.normalize(output.last_hidden_state, dim=-1)    # [B, S, E]
        past_key_values = output.past_key_values

        scores = torch.matmul(rep, self.lm)    # [B, S, V]
        next_token_logits = scores[:, -1, :]
        next_token_logits[:, self.unk] = -np.inf
        next_token = next_token_logits.max(dim=-1)[1]    # [B]
        return next_token, past_key_values
