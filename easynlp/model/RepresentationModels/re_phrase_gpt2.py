from model.utils import *


class Copyisallyouneed(nn.Module):

    def __init__(self, **args):
        super(Copyisallyouneed, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        # for name, param in self.phrase_encoder.named_parameters():
        #     if 'encoder.layer.11' not in name:
        #         param.requires_grad = False
        # print(f'[!] only the last BERT layer is fine-tuned')
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        if self.args['lang'] == 'zh':
            self.pad = self.tokenizer.pad_token_id
            self.unk = self.tokenizer.unk_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            self.pad = self.tokenizer.bos_token_id
            self.unk = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.bos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.token_embeddings = nn.Parameter(torch.randn((len(self.tokenizer), 768*2)))
        self.h_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size*2)
        )
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def get_phrase_rep(self, ids, ids_mask, pos, text):
        self.eval()
        rep = self.phrase_encoder(ids, ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        b_phrases, e_phrases, texts = [], [], []
        for rep_, pos_, text_ in zip(rep, pos, text):
            b_index = [i for i, j in pos_]
            e_index = [j for i, j in pos_]
            b_phrases.append(rep_[b_index, :])
            e_phrases.append(rep_[e_index, :])
            texts.extend(text_)
        b_phrases = torch.cat(b_phrases)
        e_phrases = torch.cat(e_phrases)
        phrases = torch.cat([self.s_proj(b_phrases), self.e_proj(e_phrases)], dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        output = self.h_proj(output)
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = self.h_proj(outputs.hidden_states[-1])
        # pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        with torch.no_grad():
            output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
            doc_bsz, seqlen, _ = output.size()
            doc_vl = dids_mask.sum(dim=-1)

        # extract the contextual embeddings
        s_rep = self.s_proj(output)    # [B, S, E]
        e_rep = self.e_proj(output)    # [B, S, E]
        start_embeddings, end_embeddings = [], []
        start_pos, end_pos = [], []
        counter = self.vocab_size
        for idx in range(doc_bsz):
            vl = doc_vl[idx]
            s = dindex_s[idx]
            e = dindex_e[idx]
            start_embeddings.append(s_rep[idx][:vl])
            end_embeddings.append(e_rep[idx][:vl])
            start_pos.append(counter + s)
            end_pos.append(counter + e)
            counter += vl
        # [B*, E]
        start_embeddings = torch.cat(start_embeddings)
        end_embeddings = torch.cat(end_embeddings)
        start_pos = torch.stack(start_pos)
        end_pos = torch.stack(end_pos)

        # extract query represetnation
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        vl = batch['vl']

        query_reps, token_labels, phrase_labels, counter = [], [], [], 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            query_reps.append(hn[:l-1])
            token_labels.append(ids_[1:l])
            pos_list_set = set(pos_list)
            for i in range(l - 1):
                if i in pos_list_set:
                    phrase_labels.append(self.vocab_size + counter)
                    counter += 1
                else:
                    phrase_labels.append(-1)
        query_reps = torch.cat(query_reps)
        start_query_reps = query_reps[:, :self.model.config.hidden_size]
        end_query_reps = query_reps[:, self.model.config.hidden_size:]
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # for token phrase mixing
        candidate_reps = torch.cat([self.token_embeddings, torch.cat([start_embeddings, end_embeddings], dim=-1)], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), token_labels] = 1.
        logits[phrase_pos_index, start_pos] = -1e3
        loss_ = F.log_softmax(logits, dim=-1) * mask
        t_loss = (-loss_.sum(dim=1)).mean()
        t_acc = logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        t_acc = t_acc.to(torch.float).mean().item()

        # for start training
        candidate_reps = torch.cat([self.token_embeddings[:, :self.model.config.hidden_size], start_embeddings], dim=0)
        logits = torch.matmul(start_query_reps, candidate_reps.t())    # [Q, B*]   
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, start_pos] = 1.
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        s_loss = (-loss_.sum(dim=-1)).sum() / valid_num
        s_acc = logits[phrase_pos_index].max(dim=-1)[1] == start_pos
        s_acc = s_acc.to(torch.float).mean().item()

        # for end training
        candidate_reps = torch.cat([self.token_embeddings[:, self.model.config.hidden_size:], end_embeddings], dim=0)
        logits = torch.matmul(end_query_reps, candidate_reps.t())    # [Q, B*]   
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, end_pos] = 1.
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        e_loss = (-loss_.sum(dim=-1)).sum() / valid_num
        e_acc = logits[phrase_pos_index].max(dim=-1)[1] == end_pos
        e_acc = e_acc.to(torch.float).mean().item()

        return t_loss, s_loss, e_loss, t_acc, s_acc, e_acc

    def get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.reshape(-1))
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == label.reshape(-1)).to(torch.long)
        valid_mask = (label != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc


