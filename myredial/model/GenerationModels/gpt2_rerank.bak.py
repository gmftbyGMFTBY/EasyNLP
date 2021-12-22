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

    def cut_from_whole_batch(self, ids, ids_mask, length):
        ids, ids_mask = deepcopy(ids), deepcopy(ids_mask)
        _, length_whole = ids.size()
        ids = ids[:, -length:]
        ids_mask = ids_mask[:, -length:]
        pad_seqlen = (ids_mask == self.pad).sum(dim=-1)    # [B]
        pad_seqlen = torch.clamp(pad_seqlen - (length_whole - length), min=0)
        ids_pos = []
        for pad_seqlen_ in pad_seqlen:
            ids_pos.append(
                [0] * pad_seqlen_ + \
                torch.arange(length - pad_seqlen_).tolist()
            )
        ids_pos = torch.LongTensor(ids_pos).cuda()
        return ids, ids_mask, ids_pos

    def gtd_loss(self, batch):
        # get the ground-truth sequence and the generated tokens
        # g: [B, S]; n: [B, S']
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, length = ids.size()
        ground_truth_length = self.args['gtd_prefix_len'] + self.args['gtd_generated_len']
        g_ids, g_ids_mask, g_pos_ids = self.cut_from_whole_batch(ids, ids_mask, ground_truth_length)
        prefix_ids = g_ids[:, :-self.args['gtd_generated_len']]
        prefix_ids_mask = g_ids_mask[:, :-self.args['gtd_generated_len']]
        prefix_pos_ids = g_pos_ids[:, :-self.args['gtd_generated_len']]

        # sample sequence
        generated_tokens = self.sample_sequence(prefix_ids, prefix_ids_mask, prefix_pos_ids)
        # construct the inputs(ids, ids_mask, label) for the detector
        gtd_ids, gtd_ids_mask, gtd_label = [], [], []
        for bi in range(bsz):
            # label is 0
            prefix_tokens = self.convert_vocab_from_gpt2_to_bert(
                [i for i in prefix_ids[bi].tolist() if i != self.pad]
            )
            gtd_ids.append(prefix_tokens + self.convert_vocab_from_gpt2_to_bert(generated_tokens[bi]))
            gtd_label.append([-100] * len(prefix_tokens) + [0] * len(generated_tokens[bi]))
            # label is 1
            gtd_ids.append(
                self.convert_vocab_from_gpt2_to_bert(
                    [i for i in g_ids[bi].tolist() if i != self.pad]
                )
            )
            gtd_label.append([-100] * len(prefix_tokens) + [1] * (len(gtd_ids[-1]) - len(prefix_tokens)))
        max_length = max([len(i) for i in gtd_ids])
        # convert list to tensor
        gtd_ids = [torch.LongTensor(i) for i in gtd_ids]
        gtd_label = [torch.LongTensor(i) for i in gtd_label]
        gtd_ids = pad_sequence(gtd_ids, batch_first=True, padding_value=self.pad)
        gtd_label = pad_sequence(gtd_label, batch_first=True, padding_value=-100)
        gtd_ids_mask = generate_mask(gtd_ids)
        gtd_ids, gtd_label, gtd_ids_mask = to_cuda(gtd_ids, gtd_label, gtd_ids_mask)
        # feedforward to the backbone of the BertModel model
        hidden = self.detector(
            input_ids=gtd_ids,
            attention_mask=gtd_ids_mask,
        ).last_hidden_state
        hidden = self.de_head(hidden)    # [B, S, 2]
        # random shuffle
        hidden = hidden.view(-1, 2)
        gtd_label = gtd_label.view(-1)
        random_index = torch.randperm(len(gtd_label))
        hidden = hidden[random_index, :]
        gtd_label = gtd_label[random_index]
        gtd_loss = self.de_loss_fct(hidden, gtd_label)
        # acc
        effective_index = (gtd_label != -100)
        chosen = hidden.max(dim=-1)[1][effective_index]
        chosen_label = gtd_label[effective_index]
        acc = (chosen == chosen_label).to(torch.float).mean().item()
        return gtd_loss, acc
    
    @torch.no_grad()
    def sample_sequence(self, ids, ids_mask, ids_pos):
        '''greedy search with batch inference, pad in the left'''
        self.model.eval()
        batch_size, seqlen = ids.size()
        generated_token = [[] for _ in range(batch_size)]
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
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated_token[idx].append(t)
            if max([len(i) for i in generated_token]) >= self.args['gtd_generated_len']:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for tokens in generated_token:
            tokens = [i for i in tokens if i not in self.special_tokens]
            rest.append(tokens)
        return rest

    @torch.no_grad()
    def _predict_fine_grained_rerank(self, prefix, candidate):
        '''prefix: [B, S]; candidate: [B, K]; return chosen_token_ids [B, 1]'''
        self.detector.eval()
        bsz, k = candidate.size()
        ids = []
        for p, cs in zip(prefix, candidate.tolist()):
            for c in cs:
                ids.append(torch.LongTensor(p + [c]))
        # [B*K, S]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.detector_vocab.pad_token_id)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        hidden = self.detector(ids, ids_mask).last_hidden_state
        score = F.softmax(self.de_head(hidden), dim=-1)[:, -1, 1]    # [B*K]
        score = torch.stack(torch.split(score, k))    # [B, K]
        max_score = score.max(dim=-1)[1]    # [B]
        # build the chosen tokens
        chosen_tokens = []
        for s, c in zip(max_score, candidate):
            chosen_tokens.append(c[s])
        return torch.LongTensor(chosen_tokens).cuda().unsqueeze(-1)

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
        # build the prefix w/o. the padding tokens
        prefix = [[t for t in tokens.tolist() if t != self.pad] for tokens in ids]
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

            # coarse-grained recall
            next_token_candidate = next_token_logits.topk(self.args['gray_cand_num'], dim=-1)[1]    # [B, K]
            # fine-grained rerank
            chosen_token = self._predict_fine_grained_rerank(prefix, next_token_candidate)    # [B, 1]
            for idx, t in enumerate(chosen_token.squeeze(-1).tolist()):
                generated[idx].append(t)
                # prefix appending
                prefix[idx].append(t)
            if max([len(i) for i in generated]) >= self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = chosen_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            pbar.update(1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

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

        # GTD loss (generated tokens detection)
        gtd_loss, gtd_acc = self.gtd_loss(batch)
        return loss, gtd_loss, gen_acc, gtd_acc
