from model.utils import *


class WriterGPT2Model(nn.Module):

    '''only for inference'''

    def __init__(self, **args):
        super(WriterGPT2Model, self).__init__()
        model = args['gpt2_pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = BertTokenizer.from_pretrained(model)
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.vocab.convert_tokens_to_ids('[PAD]'))
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.special_tokens = set([self.unk, self.pad, self.cls, self.sep])
        self.topk = args['gpt2_topk']
        self.topp = args['gpt2_topp']
        self.test_max_len = args['gpt2_gen_max_len']
        self.test_max_ctx_len = args['gpt2_gen_max_ctx_len']
        self.corrupt_min_ratio = args['corrupt_min_ratio']
        self.corrupt_max_ratio = args['corrupt_max_ratio']
        self.corrupt_min_topk = args['corrupt_min_topk']
        self.corrupt_max_topk = args['corrupt_max_topk']
        self.total_step = args['total_step']
        self.args = args

    def forward(self, batch, current_step):
        '''batch inference, pad in the left'''
        cids = deepcopy(batch['gpt2_cids'])
        cids_mask = deepcopy(batch['gpt2_cids_mask'])
        cids_pos = deepcopy(batch['gpt2_pos_ids'])
        cids_o = deepcopy(cids)
        batch_size, seqlen = cids.size()
        generated = [[] for _ in range(batch_size)]
        over_flag = [0] * batch_size
        past_key_values = None
        while True:
            output = self.model(
                input_ids=cids,
                attention_mask=cids_mask,
                position_ids=cids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf

            self.corrupt(next_token_logits, current_step)
            next_token_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1),
                num_samples=1,
            )    # [B, 1]
            # save the flag
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            cids = next_token
            cids_mask = torch.ones_like(cids)
            cids_pos = 1 + cids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def corrupt(self, logits, current_step):
        '''logits: [B, V], the formualtion is:
            y = (y_{min} - y_{max})/total_step * current_step + y_{max}'''
        current_corrupt_ratio = (self.corrupt_min_ratio-self.corrupt_max_ratio)/self.total_step * current_step + self.corrupt_max_ratio
        current_corrupt_topk  = (self.corrupt_min_topk - self.corrupt_max_topk)/self.total_step * current_step + self.corrupt_max_topk
        current_corrupt_topk = int(current_corrupt_topk)
        if random.random() < current_corrupt_ratio: 
            topk_idx = torch.topk(logits, current_corrupt_topk)[1]
            logits[torch.arange(len(logits)).unsqueeze(1), topk_idx] = -1e3
        return logits
