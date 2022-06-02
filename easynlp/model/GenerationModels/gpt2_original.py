from model.utils import *
from .utils import *


class GPT2OriginalModel(nn.Module):

    def __init__(self, **args):
        super(GPT2OriginalModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.vocab)
        self.args = args

        if args['lang'] == 'en':
            self.pad = self.vocab.eos_token_id
            self.unk = self.vocab.unk_token_id
            self.special_tokens = set([self.pad])
        else:
            self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
            self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.topk = args['topk']
        self.topp = args['topp']
        
        self.test_max_len = args['test_max_len']
        self.test_max_ctx_len = args['test_max_ctx_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask):
        self.model.eval()
        ids, ids_mask, label = ids[:, :-1], ids_mask[:, :-1], ids[:, 1:]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        logits = output.logits
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
        ppl = math.exp(loss.item())
        time.sleep(0.2)
        return ppl

    @torch.no_grad()
    def beam_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids,
            prefix_length+self.test_max_len,
            num_beams=self.args['num_beam'],
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id
        )
        beam_output = beam_output[:, prefix_length:]
        string = ''.join(self.vocab.convert_ids_to_tokens(beam_output[0]))
        return string

    @torch.no_grad()
    def greedy_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
            )[0]    # [1, S, V]
            next_token_logits = output[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(0)
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            ids = ids[:, -self.test_max_ctx_len:]
        string = self.vocab.decode(generated)
        return string

    @torch.no_grad()
    def contrastive_search(self, batch):
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        for step in range(self.test_max_len):
            input_ids = ContrastiveDecodingOneStep(
                self.model,
                input_ids,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.unk
            )
        # input_ids contains the prefix, cut it
        input_ids = input_ids[:, prefix_length:]
        string = ''.join(self.vocab.convert_ids_to_tokens(input_ids[0]))
        return string

    @torch.no_grad()
    def topk_topp_search(self, batch):
        '''batch_size is 1'''
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids, 
            prefix_length+self.test_max_len, 
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            top_p=self.topp,
            top_k=self.topk,
            do_sample=True,
        )
        beam_output = beam_output[:, prefix_length:]
        string = self.vocab.decode(beam_output[0])
        return string

    def forward(self, batch):
        ids, ids_mask = batch['ids'], batch['ids_mask']
        ids, ods = ids[:, :-1], ids[:, 1:]
        ids_mask = ids_mask[:, :-1]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = output.logits
        loss = self.gen_loss_fct(
            gen_logits.view(-1, gen_logits.size(-1)), 
            ods.reshape(-1)
        )
        # token acc
        chosen_tokens = torch.max(gen_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == ods.reshape(-1)).to(torch.long)
        valid_mask = (ods != self.pad).reshape(-1)
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
    def calculate_ppl(self, ids, ids_mask):
        ids, ids_mask, ods = ids[:, :-1], ids_msak[:, :-1], ids[:, 1:]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
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
