from model.utils import *
from .utils import *


class GPT2OriginalModel(nn.Module):

    def __init__(self, **args):
        super(GPT2OriginalModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.args = args
        self.topk = args['topk']
        self.topp = args['topp']

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, pos_ids, label):
        output = self.model(input_ids=ids, attention_mask=ids_mask, position_ids=pos_ids)
        logits = output.logits
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
        return math.exp(loss.item())

    @torch.no_grad()
    def predict(self, batch):
        '''contrastive search'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]

        past_key_values = None
        last_hidden_states = None
        first_step = 0
        logits = None
        for step in range(self.test_max_len):
            ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepBatch(
                self.model,
                ids,
                ids_mask,
                ids_pos,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.args['sampling_probability'],
                self.pad,
                min(1., (step+1)/self.args['sep_smooth_length']),
                past_key_values,
                last_hidden_states,
                self.vocab,
                logits,
                first_step=first_step == 0,
            )
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            ids_mask = torch.ones_like(ids)
            first_step += 1
            # collect ids: [B, 1]
            tokens = ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
        # ignore the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    @torch.no_grad()
    def _predict(self, batch):
        '''topk-topp search with batch inference, pad in the left'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        step = 0
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
            next_token_logits[:, self.sep] /= min(1.0, (step+1)/self.args['sep_smooth_length']) 
            filtered_logits = top_k_top_p_filtering_batch(
                next_token_logits,
                top_k=self.topk,
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1
            )
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            step += 1
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ods = batch['ods']
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = output.logits
        loss = self.gen_loss_fct(
            gen_logits.view(-1, gen_logits.size(-1)), 
            ods.view(-1)
        )
        # token acc
        chosen_tokens = torch.max(gen_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == ods.view(-1)).to(torch.long)
        valid_mask = (ods != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc


class GPT2DialogModel(nn.Module):

    def __init__(self, **args):
        super(GPT2DialogModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.args = args
        self.topk = args['topk']
        self.topp = args['topp']

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, pos_ids, label):
        output = self.model(input_ids=ids, attention_mask=ids_mask, position_ids=pos_ids)
        logits = output.logits
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
        return math.exp(loss.item())

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]

        past_key_values = None
        last_hidden_states = None
        logits = None
        sampling_prefix_len = self.args['sampling_prefix_len']
        for step in range(self.test_max_len):
            ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepBatch(
                self.model,
                ids,
                ids_mask,
                ids_pos,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.sep,
                min(1., (step+1)/self.args['sep_smooth_length']),
                past_key_values,
                last_hidden_states,
                self.vocab,
                logits,
                step,
                step < self.args['sampling_prefix_len'],
            )
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            ids_mask = torch.ones_like(ids)
            # collect ids: [B, 1]
            tokens = ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
        # batch size is 1
        return generated

    @torch.no_grad()
    def _predict(self, batch):
        '''topk-topp search with batch inference, pad in the left'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        step = 0
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
            next_token_logits[:, self.sep] /= min(1.0, (step+1)/self.args['sep_smooth_length']) 
            filtered_logits = top_k_top_p_filtering_batch(
                next_token_logits,
                top_k=self.topk,
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1
            )
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            step += 1
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = output.logits
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
        return loss, gen_acc
