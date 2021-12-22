from model.utils import *
from .utils import *


class GPT2UNModel(nn.Module):

    def __init__(self, **args):
        super(GPT2UNModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.sample_token_num = args['sample_token_num']
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
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
            output = self.model(
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

    def token_unlikelyhood(self, ids, logits):
        # slow but more effective
        # ids: [B, S], logits: [B, S, V], remvove the the last token
        # sub_logits = F.softmax(logits, dim=-1)    # [B, S, V]
        # loss = []
        # bsz, seqlen = ids.size()
        # for i in range(bsz):
        #     sampled_index = random.sample(range(seqlen), self.sample_token_num)
        #     for j in sampled_index:
        #         if ids[i,j].item() == self.pad:
        #             continue
        #         candidates = list(set(ids[i, :j].tolist()))
        #         loss.append((-torch.log(1e-3 + 1 - sub_logits[i, j, candidates])).sum())
        # loss = torch.stack(loss).mean()
        # return loss

        # fast mode but not effective
        logits = logits[:, :-1, :]
        target = ids[:, 1:]
        bsz, seqlen = target.size()
        logits = F.softmax(logits, dim=-1)    # [B, S, V]
        cands = target.unsqueeze(1).expand(-1, target.size(-1), -1)    # [B, S, S]
        cands = cands.tril(-1)    # [B, S, S]

        # donot include it self
        cands = cands.masked_fill(cands == target.unsqueeze(2), self.pad)
        negative_cands = torch.zeros_like(logits).scatter(2, cands, 1)    # [B, S, V]
        loss = -torch.log(1e-5 + 1 - logits) * negative_cands    # [B, S, V]
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']

        batch_size = ids.shape[0]
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits

        # generative loss
        # gen_logits: [B, S, V]; label: [B, S]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        
        # unlikelyhood loss
        un_loss = self.token_unlikelyhood(ids, gen_logits)
        loss += un_loss
        return loss, gen_acc
