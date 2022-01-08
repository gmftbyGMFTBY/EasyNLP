from model.utils import *
from .utils import *


class GPT2UnlikelyModel(nn.Module):

    def __init__(self, **args):
        super(GPT2UnlikelyModel, self).__init__()
        self.model = GPT2IteractiveLMHeadModel.from_pretrained(args['pretrained_model'])
        self.cls_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 1),
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.cls_loss_fct = nn.BCEWithLogitsLoss()
        self.vocab = BertTokenizerFast.from_pretrained(args['pretrained_model'])
        self.pad, self.cls, self.unk, self.sep = self.vocab.convert_tokens_to_ids(['[PAD]', '[CLS]', '[UNK]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.test_max_len = args['gen_max_len']
        self.test_max_ctx_len = args['gen_max_ctx_len']
        self.repetition_penalty = args['repetition_penalty']
        self.iteractive_num = args['iteractive_num']
        self.scale_ratio = args['scale_ratio']

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits, _ = self.gpt2_forward(ids, ids_mask, torch.zeros(len(ids), 768).cuda())
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    def calculate_token_acc(self, shift_logits, shift_labels): 
        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return gen_acc

    def get_lm_loss(self, shift_logits, shift_labels):
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        return loss
    
    @torch.no_grad()
    def predict_once(self, batch, iteractive_embd):
        '''batch_size is 1'''
        ids = batch['cids']
        ids_mask = batch['cids_mask']
        generated = []
        while True:
            logits, hidden = self.gpt2_forward(ids, ids_mask, iteractive_embd)
            # cls rest and cls hidden for next iteraction
            cls_output = hidden[0, -1, :]    # [E]
            cls_rest = torch.sigmoid(self.cls_head(cls_output))
            # 
            next_token_logits = logits[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            if next_token == self.sep or len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            ids = ids[:, -self.test_max_ctx_len:]
            ids_mask = torch.ones_like(ids)
        # make sure the 1 batch size is append to the cls_output
        return generated, cls_output.unsqueeze(0), cls_rest.item()

    @torch.no_grad()
    def predict(self, batch):
        ids = batch['cids']
        ids_mask = batch['cids_mask']
        cls_output = F.normalize(torch.randn(len(ids), 768).cuda(), dim=-1)
        best_cls_rest = -1
        cls_rest_history = []
        for _ in range(self.iteractive_num):
            n_generated, cls_output, cls_rest = self.predict_once(
                batch,
                cls_output,
            )
            cls_rest_history.append(cls_rest)
            if cls_rest > best_cls_rest:
                generated = n_generated
                best_cls_rest = cls_rest
            # during predict, not add the additional noise into the cls_output for the accurate prediction
            cls_output = F.normalize(cls_output, dim=-1)
        return generated

    def gpt2_forward(self, ids, ids_mask, iteractive_hidden_states):
        # lower the influence of this noise or iteractive embedding
        iteractive_hidden_states *= self.scale_ratio
        output = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            iteractive_hidden_states=iteractive_hidden_states,
            output_hidden_states=True,
        )
        gpt2_hidden_states = output.hidden_states[-1]
        gpt2_logits = output.logits
        return gpt2_logits, gpt2_hidden_states

    def forward(self, batch):
        # ===== input ===== #
        noise_ids      = batch['noise_ids']
        noise_mask     = batch['noise_mask']
        cls_label      = batch['cls_label']     # [B]
        last_token_pos = batch['last_token_pos']
        # ground-truth
        gpt2_ids   = batch['gpt2_ids']
        gpt2_mask  = batch['gpt2_mask']
        batch_size = len(gpt2_ids)
        # ===== input ===== # 

        # input the first stage ids, only inference
        with torch.no_grad():
            cls_output = F.normalize(
                torch.randn(batch_size, 768).cuda(), 
                dim=-1
            )
            _, hidden = self.gpt2_forward(
                noise_ids,
                noise_mask,
                cls_output,
            )
        # cls loss
        cls_output = hidden[range(batch_size), last_token_pos, :]
        cls_rest = self.cls_head(cls_output).squeeze(-1)
        cls_loss = self.cls_loss_fct(cls_rest, cls_label.to(torch.float))

        # input the second stage ids and last iteractive hidden states
        cls_otuput = F.normalize(cls_output, dim=-1)
        logits, _ = self.gpt2_forward(
            gpt2_ids, 
            gpt2_mask, 
            cls_output,
        )
        # generation loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = gpt2_ids[..., 1:].contiguous()
        gpt2_token_acc = self.calculate_token_acc(
            shift_logits, 
            shift_labels
        )
        gpt2_loss = self.get_lm_loss(
            shift_logits, 
            shift_labels
        )
        loss = gpt2_loss + cls_loss
        return loss, gpt2_token_acc, math.exp(gpt2_loss.item())
