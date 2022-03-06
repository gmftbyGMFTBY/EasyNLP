from model.utils import *
from .utils import *


class GPT2TokenRerankModel(nn.Module):

    def __init__(self, **args):
        super(GPT2TokenRerankModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)

        self.token_reranker = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 2)
        )

        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.test_max_len = args['test_max_len']
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.args = args
        self.topk = args['topk']
        self.topp = args['topp']

        self.criterion = nn.CrossEntropyLoss()
        self.iter_hn_num = args['iter_hn_num']

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

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        bsz, seqlen = ids.size()
        length = ids_mask.sum(dim=-1).tolist()    # [B]
        with torch.no_grad():
            # output: [B, S, E]
            output = self.model(
                input_ids=ids, 
                attention_mask=ids_mask,
                output_hidden_states=True,    
            )
            gen_logits = output.logits
            gen_hidden = output.hidden_states[-1]    # [B, S, E]
            vocab_size = gen_logits.size(-1)
            filtered_logits = top_k_top_p_filtering_batch(
                gen_logits.reshape(-1, vocab_size),
                top_k=self.topk,
                top_p=self.topp,
            ) 
            filtered_logits = filtered_logits.reshape(bsz, seqlen, vocab_size)
            # mask the ground-truth
            mask_label = torch.cat([ids[:, 1:], torch.zeros(bsz, 1).cuda()], dim=-1).to(torch.long)    # [B, S]
            value = torch.zeros_like(filtered_logits)
            value.fill_(-1000)
            filtered_logits = filtered_logits.scatter(2, mask_label.unsqueeze(2), value)
        # da augmentation, mask the ground-truth
        next_token = torch.multinomial(
            F.softmax(filtered_logits.reshape(-1, vocab_size), dim=-1),
            num_samples=1
        ).reshape(bsz, seqlen)    # [B, S]
        loss, acc, counter = 0, 0, 0
        for _ in range(self.iter_hn_num):
            # sample the token for each instance
            sample_index = torch.LongTensor([random.choice(range(l-1)) for l in length]).cuda()
            sample_token = next_token[range(bsz), sample_index]    # [B]
            # replace the original token
            new_ids = ids.clone()
            new_ids[range(bsz), sample_index + 1] = sample_token
            # build the new attention mask
            new_mask = ids_mask.clone()
            new_mask[range(bsz), sample_index + 1] = 0.
            # feed again
            with torch.no_grad():
                output = self.model(
                    input_ids=new_ids, 
                    attention_mask=new_mask,
                    output_hidden_states=True,    
                )
                new_gen_hidden = output.hidden_states[-1][range(bsz), sample_index + 1, :]    # [B, 768]
            new_gen_logits_hn = self.token_reranker(new_gen_hidden)    # [B, 2]

            # positive samples (ground-truth)
            new_gen_logits_po = self.token_reranker(gen_hidden[range(bsz), sample_index, :])    # [B, 2]

            # build the training instances
            label = torch.LongTensor([0] * bsz + [1] * bsz).cuda()    # [B*2]
            instance = torch.cat([new_gen_logits_hn, new_gen_logits_po], dim=0)    # [B*2, 2]    

            # random shuffle
            random_index = list(range(bsz * 2))
            random.shuffle(random_index)
            label = label[random_index]
            instance = instance[random_index, :]

            loss += self.criterion(instance, label)
            acc += (instance.max(dim=-1)[1] == label).sum().item()
            counter += len(label)
        acc /= counter
        return loss, acc
