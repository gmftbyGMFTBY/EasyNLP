from model.utils import *

class GPT2CLEncoder(nn.Module):

    def __init__(self, **args):
        super(GPT2CLEncoder, self).__init__()
        model = args['pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)

        # special tokens
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls])
        self.model = GPT2CLHeadModel(model, len(self.vocab), unk=self.unk, pad=self.pad, temp=args['temp'])
        self.test_max_len = args['test_max_len']
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        rep = self.model.model(
            input_ids=ids, 
            attention_mask=ids_mask
        ).last_hidden_state
        gen_logits = torch.matmul(rep, self.model.lm)    # [B, S, V]
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
            next_token, past_key_values = self.model.predict_one_step(
                ids,
                ids_mask,
                ids_pos,
                past_key_values,
            )
            for idx, t in enumerate(next_token.tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token.unsqueeze(1)    # [B, 1]
            ids_mask = torch.ones_like(ids)    # [B, 1]
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        loss, token_acc = self.model(gpt2_ids, gpt2_ids_mask)
        return loss, token_acc
