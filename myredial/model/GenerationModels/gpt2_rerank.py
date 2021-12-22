from model.utils import *
from .utils import *


class GPT2RerankModel(nn.Module):

    def __init__(self, **args):
        super(GPT2RerankModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.sequence_ngram_n = args['sequence_ngram_n']
        self.sample_token_num = args['sample_token_num']
        self.topk_upper, self.topk_lower = args['cg_topk_upper'], args['cg_topk_lower']
        self.cg_alpha = args['cg_alpha']
        self.fg_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 2)
        )
        self.fg_loss_fct = nn.CrossEntropyLoss()
        self.args = args

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
        '''batch inference with fine-grained rerank, pad in the left'''
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
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(-1)

            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) >= self.test_max_len:
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
    
    def repetition_token_hard_negative(self, ids, logits):
        # ids: [B, S], logits: [B, S, V], remvove the the last token
        sub_logits = F.softmax(logits, dim=-1)    # [B, S, V]
        loss = []
        bsz, seqlen = ids.size()
        for i in range(bsz):
            sampled_index = random.sample(range(seqlen), self.sample_token_num)
            for j in sampled_index:
                if ids[i,j].item() == self.pad:
                    continue
                # candidates
                candidates = list(set(ids[i, :j].tolist()))
                loss.append((-torch.log(1e-3 + 1 - sub_logits[i, j, candidates])).sum())
        loss = torch.stack(loss).mean()
        return loss

    def embedding_hard_negative(self, ids, logits):
        '''ids: [B, S]; logits: [B, S, V]'''
        target, sub_logits = ids[:, 1:], logits[:, :-1, :]
        sub_logits = F.softmax(sub_logits, dim=-1)
        candidates, candidates_index = sub_logits.topk(self.topk_upper, dim=-1)
        # [B, S, K]
        candidates, candidates_index = candidates[:, :, self.topk_lower:], candidates_index[:, :, self.topk_lower:]
        # ignore the ground-truth
        candidates = candidates.masked_fill(candidates_index == target.unsqueeze(-1), 1e-5)    # [B, S, K]
        candidates = candidates.reshape(-1, candidates.size(-1))    # [B*S, K]
        loss = -torch.log(torch.clamp(1 - candidates, min=1e-5))
        loss = self.cg_alpha * loss.sum(dim=-1).mean()
        return loss

    def train_generation_token_detection(self, batch):
        ## prepare inputs
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, _ = ids.size()
        # ids: [B, S]; ids_mask: [B, S]; ids_pos: [B, S]
        ids = ids[:, -self.args['gtd_prefix_len']:]
        ids_mask = ids_mask[:, -self.args['gtd_prefix_len']:]
        pad_seqlen = (ids_mask == self.pad).sum(dim=-1)    # [B]
        pad_seqlen = torch.clamp(pad_seqlen - (length - self.args['gtd_prefix_len']), min=0)
        ids_pos = []
        for pad_seqlen_ in pad_seqlen:
            ids_pos.append(
                [0] * pad_seqlen_ + \
                torch.arange(self.args['gtd_prefix_len'] - pad_seqlen_).tolist()
            )
        ids_pos = torch.LongTensor(ids_pos).cuda()

        
        gt_ids = ids[:, -1]
        ids, ids_mask,ids_pos = ids[:, :-1], ids_mask[:, :-1], ids_pos[:, :-1]
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=ids_pos,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        hidden, gen_logtis = output.hidden_states[-1], output.logits
        gt_hidden = hidden[:, -1, :]
        gen_hidden = hidden[:, -2, :]    # [B, E]
        # hard_candidates_index = gen_logits.topk(dim=-1, self.args['fg_topk_upper'])[1]    # [B, K]
        hard_candidates_index = hard_candidates_index[:, self.args['fg_topk_lower']:]
        candidates = gen_hidden[hard_candidates_index != gt_ids.unsqueeze(1)]


    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        pos_ids = batch['pos_ids']

        batch_size = ids.shape[0]
        gen_logits = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            position_ids=pos_ids
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
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        
        # coarse-grained loss
        cg_loss = self.embedding_hard_negative(ids, gen_logits)
        # cg_loss += self.repetition_token_hard_negative(ids, gen_logits)
        
        # fine-grained loss
        # fg_loss = self.generation_token_detection(ids, ids_mask, pos_ids)
        return loss, cg_loss, gen_acc
