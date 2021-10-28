from model.utils import *
from .utils import *


class MyGPT2Model(nn.Module):

    def __init__(self, **args):
        super(MyGPT2Model, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        # pad token is 0
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.test_max_len = args['gen_max_len']
        self.test_max_ctx_len = args['gen_max_ctx_len']
        self.repetition_penalty = args['repetition_penalty']

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
        '''batch_size is 1'''
        ids = batch['cids']
        ids_mask = batch['cids_mask']
        # ids: [1, S]; ids_mask: [1, S]
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
                attention_mask=ids_mask,
            )[0]    # [1, S, V]
            next_token_logits = output[-1, -1, :]    # [V]
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
        return generated

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['mask']

        batch_size = ids.shape[0]
        # [B, S, V]
        gen_logits = self.model(
            input_ids=ids, attention_mask=ids_mask
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
        return loss, gen_acc, math.exp(loss.item())
