from model.utils import *
from model.AugmentationModels.bert_mask_for_rerank import *
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
        self.cg_sample_ratio = args['cg_sample_ratio']
        self.cg_alpha = args['cg_alpha']
        self.fg_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 2)
        )
        self.fg_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.rerank_k = args['rerank_k']
        self.args = args

        # train mode
        if args['mode'] == 'train':
            self.aug_bert_model = BERTMaskAugmentationForRerankModel(**args)

        self.predict = self.predict_with_rerank

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
    def _predict_rerank(self, o_candidates, past_key_values, position_ids):
        '''candidates: [B, K]'''
        self.model.eval()
        rest = []
        for i in range(self.rerank_k):
            candidates = o_candidates[:, i].unsqueeze(-1)
            mask = torch.ones_like(candidates)
            pos_ids = position_ids[:, -1].unsqueeze(dim=-1) + 1
            output = self.model(
                input_ids=candidates,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            hidden = F.softmax(
                self.fg_head(output.hidden_states[-1][:, 0, :]), dim=-1
            )[:, 1]   # [B]
            rest.append(hidden)
        rest = torch.stack(rest).t()    # [B, K]
        # greedy search
        chosen_idx = rest.max(dim=-1)[1]    # [B]
        chosen_token = o_candidates[range(len(o_candidates)), chosen_idx]   # [B]
        return chosen_token
    
    @torch.no_grad()
    def predict_with_rerank(self, batch):
        '''batch inference with fine-grained rerank and greedy search, pad in the left'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        pbar = tqdm(total=self.test_max_len)
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
            candidates_score, candidates = next_token_logits.topk(self.rerank_k, dim=-1)    # [B, K]
            next_token = self._predict_rerank(candidates, past_key_values, ids_pos).unsqueeze(-1)
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) >= self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            pbar.update(1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest
    
    @torch.no_grad()
    def predict(self, batch):
        '''batch inference with greedy search, pad in the left'''
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
    
    def inner_token_hard_negative(self, ids, logits, ids_mask):
        # ids/ids_mask: [B, S], logits: [B, S, V], remvove the the last token
        sub_logits, target, target_ids_mask = logits[:, :-1, :], ids[:, 1:], ids_mask[:, 1:]
        bsz, seqlen, vsz = sub_logits.size()
        sub_logits = F.softmax(sub_logits, dim=-1)    # [B, S, V]
        cands = target.unsqueeze(1).expand(-1, target.size(-1), -1)    # [B, S, S]
        # donot include it self
        cands = cands.masked_fill(cands == target.unsqueeze(2), self.pad)
        negative_cands = torch.zeros_like(sub_logits).scatter(2, cands, 1).to(torch.long)    # [B, S, V]
        # ignore the padding tokens
        padding_mask = target_ids_mask.unsqueeze(-1).expand(-1, -1, vsz)
        negative_cands = negative_cands & padding_mask
        # only update partial tokens
        # if penalty full negative samples, the language model will be harmed
        ignore_mask = torch.rand_like(negative_cands.to(torch.float16))    # [B, S, V]
        negative_cands = torch.where(
            torch.logical_and(
                ignore_mask < self.cg_sample_ratio,
                negative_cands.to(torch.bool)
            ),
            negative_cands,
            0
        )
        # penalty loss
        loss = -torch.log(torch.clamp(1 - sub_logits, min=1e-5)) * negative_cands    # [B, S, V]
        loss = self.cg_alpha * loss.sum(dim=-1).mean()
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

    def fg_train(self, batch):
        # mlm augmentation
        ids, ids_mask, label = self.aug_bert_model(batch, self.vocab)
        hidden = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        logits = self.fg_head(hidden)    # [B, S, 2]
        logits = logits.view(-1, 2)
        label = label.view(-1)
        loss = self.fg_loss_fct(logits, label)
        # acc, only counting the negative samples
        acc = (logits.max(dim=-1)[1] == label)[label != -1].to(torch.float).mean().item()
        return loss, acc

    def forward(self, batch):
        if batch['mode'] == 'coarse-grained':
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
            cg_loss = self.inner_token_hard_negative(ids, gen_logits, ids_mask)
            return loss, cg_loss, gen_acc
        elif batch['mode'] == 'fine-grained':
            fg_loss, acc = self.fg_train(batch)
            return fg_loss, acc
        else:
            raise Exception(f'[!] Unknown training mode: {batch["mode"]}')
