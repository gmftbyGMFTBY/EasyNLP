from model.utils import *
from .utils import *


class GPT2UNSeqModel(nn.Module):

    '''sequence-level unlikelyhood training: 
        1. https://arxiv.org/pdf/1908.04319.pdf
        2. https://github.com/facebookresearch/unlikelihood_training/blob/main/custom/gpt2/run_gpt2.py'''

    def __init__(self, **args):
        super(GPT2UNSeqModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = AutoTokenizer.from_pretrained(model)
        # wikitext103 all the special tokens are the same
        self.unk, self.pad, self.cls, self.sep = [self.vocab.eos_token_id] * 4
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.sequence_ngram_n = args['sequence_ngram_n']
        self.sample_ratio = args['sample_ratio']
        self.args = args

    def ngram_repeat_mask(self, xs, n):
        '''mask is 1 denotes that the ngram is the repeatition of the previous tokens in the sequence (prefix or generated tokens)'''
        mask = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            seen = set()
            xl = x.tolist()
            for j in range(len(x)-n):
                ng = tuple(xl[j:j+n])
                if ng in seen:
                    mask[i, j:j+n] = 1
                seen.add(ng)
        # [B, S]
        return mask 

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, pos_ids, label):
        self.model.eval()
        gen_logits = self.model(input_ids=ids, position_ids=pos_ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl
    
    def sample_sequence(self, batch):
        '''greedy search with batch inference, pad in the left'''
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        bsz, length = ids.size()

        # cut the long context
        ids = ids[:, -self.args['un_prefix_len']:]
        ids_mask = ids_mask[:, -self.args['un_prefix_len']:]
        pad_seqlen = (ids_mask == self.pad).sum(dim=-1)    # [B]
        pad_seqlen = torch.clamp(pad_seqlen - (length - self.args['un_prefix_len']), min=0)
        ids_pos = []
        for pad_seqlen_ in pad_seqlen:
            ids_pos.append(
                [0] * pad_seqlen_ + \
                torch.arange(self.args['un_prefix_len'] - pad_seqlen_).tolist()
            )
        ids_pos = torch.LongTensor(ids_pos).cuda()

        batch_size, seqlen = ids.size()
        generated_token = [[] for _ in range(batch_size)]
        generated = []
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
            next_token_logits = logits[:, -1, :]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            generated.append(logits[:, -1, :])
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated_token[idx].append(t)
            if len(generated) >= self.args['un_generated_len']:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # [B, S]; [B, S, V]
        return torch.LongTensor(generated_token).cuda(), torch.stack(generated).permute(1, 0, 2)

    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        self.model.eval()
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

    def sequence_unlikelyhood(self, batch):
        # sample the squence
        gen_tokens, gen_logits = self.sample_sequence(batch)
        ngram_repeat_mask = self.ngram_repeat_mask(gen_tokens, self.sequence_ngram_n)    # [B, S]
        # sequence unlikelyhood traininig loss
        gen_logits = F.softmax(gen_logits, dim=-1)    # [B, S, V]
        gen_logits = gen_logits.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)    # [B, S]
        loss = -torch.log(torch.clamp(1 - gen_logits, min=1e-5)) * ngram_repeat_mask    # [B, S]
        loss = loss.sum()
        effective_tokens_num = torch.sum(ngram_repeat_mask).item()
        if effective_tokens_num > 0:
            loss /= effective_tokens_num
        return loss
    
    def token_unlikelyhood(self, ids, logits, ids_mask):
        logits = logits[:, :-1, :]
        target = ids[:, 1:]
        target_ids_mask = ids_mask[:, 1:]
        bsz, seqlen, vsz = logits.size()
        logits = F.softmax(logits, dim=-1)    # [B, S, V]
        cands = target.unsqueeze(1).expand(-1, target.size(-1), -1)    # [B, S, S]
        pad_matrix = torch.zeros_like(cands).to(torch.long)
        pad_matrix.fill_(self.pad)
        cands = cands.tril(-1) + pad_matrix.triu()
        # donot include it self
        cands = cands.masked_fill(cands == target.unsqueeze(2), self.pad)
        negative_cands = torch.zeros_like(logits).scatter(2, cands, 1).to(torch.long)    # [B, S, V]
        # ignore the padding tokens
        padding_mask = target_ids_mask.unsqueeze(-1).expand(-1, -1, vsz)
        negative_cands = negative_cands & padding_mask
        # only update partial tokens
        # ignore_mask = torch.rand_like(negative_cands.to(torch.float16))
        # negative_cands = torch.where(
        #     torch.logical_and(
        #         ignore_mask < self.sample_ratio,
        #         negative_cands.to(torch.bool)
        #     ),
        #     negative_cands, 0
        # )
        loss = -torch.log(torch.clamp(1 - logits, min=1e-5)) * negative_cands    # [B, S, V]
        return loss.sum(dim=-1).mean()

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        pos_ids = batch['pos_ids']

        batch_size = ids.shape[0]
        gen_logits = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            position_ids=pos_ids,
        )
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
        valid_mask = (shift_labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        if batch['token_un'] is True:
            # token-level unlikelyhood loss
            un_loss = self.token_unlikelyhood(ids, gen_logits, ids_mask)
        else:
            # sequence-level unlikelyhood loss
            un_loss = self.sequence_unlikelyhood(batch)
        return loss, un_loss, gen_acc
