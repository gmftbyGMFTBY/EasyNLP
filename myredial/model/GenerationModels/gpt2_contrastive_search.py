from model.utils import *
from .utils import *

class ContrastiveGPT2Encoder(nn.Module):

    '''contrastive search for gpt2 model.
    For inference, please load the model into the gpt2 model (model_name in the GenerationModels)'''

    def __init__(self, **args):
        super(ContrastiveGPT2Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if args['dataset'] in ['writer-rank']:
            self.pad = self.tokenizer.pad_token_id
            self.special_tokens = set([self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        else:
            self.pad = self.tokenizer.bos_token_id
            self.special_tokens = set([self.tokenizer.bos_token_id])
        self.vocab_size = len(self.tokenizer)

        # model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        # decoding length
        self.test_max_len = args['test_gen_max_len']
        # ignore the pad_token (-100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.margin = args['margin']

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
                self.tokenizer,
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

    def forward(self, batch):
        input_ids, ids_mask, labels = batch['ids'], batch['ids_mask'], batch['ods']
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask, output_hidden_states=True)
        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        # token_acc
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.view(-1) == labels.view(-1)).to(torch.long)
        valid_mask = (labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(self.margin, cosine_scores, input_ids, self.pad, prefix_len=0)
        return mle_loss, gen_acc, cl_loss

    def calculate_ppl(self, input_ids, ids_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask)
        logits = outputs.logits
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        return math.exp(mle_loss.item())
