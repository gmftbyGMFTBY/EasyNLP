from model.utils import *

class DensePhraseEncoder(nn.Module):

    '''contrastive search for gpt2 model.
    For inference, please load the model into the gpt2 model (model_name in the GenerationModels)'''

    def __init__(self, **args):
        super(DensePhraseEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'])
        self.pad = self.bert_tokenizer.pad_token_id
        self.sep = self.bert_tokenizer.sep_token_id
        self.hn_num = self.args['hard_neg_for_each_doc']

        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2Model.from_pretrained(model_name)
        self.s_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)        
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)        
        )

    @torch.no_grad()
    def get_phrase_rep(self, ids, ids_mask, pos, text):
        self.eval()
        rep = self.phrase_encoder(ids, ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        phrases, texts = [], []
        for rep_, pos_, text_ in zip(rep, pos, text):
            for (b, e) in pos_:
                b_rep = rep_[b, :]
                e_rep = rep_[e, :]
                p_rep = torch.cat([b_rep, e_rep], dim=-1)
                phrases.append(p_rep)
            texts.extend(text_)
        phrases = torch.stack(phrases)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        rep = self.model(input_ids=ids).last_hidden_state
        query = torch.cat([
            self.s_proj(rep[:, -1, :]),
            self.e_proj(rep[:, -1, :])
        ], dim=-1)
        query = F.normalize(query, dim=-1)
        return query

    def forward(self, batch):
        ## generation fine-tuning gpt2
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask)
        last_hidden_states = outputs.last_hidden_state

        # phrase begin position
        pos_index = batch['pos_ids']    # [B_p]
        pos_index_end = batch['pos_ids_end']
        ext_last_hidden_states = []
        hard_ext_last_hidden_states = []

        repetition_penalty_query = []
        length = ids_mask.sum(dim=-1)
        for hs, pos_list, pos_end_list, l in zip(last_hidden_states, pos_index, pos_index_end, length):
            # hs: [S, E]
            ext_last_hidden_states.append(hs[pos_list, :])
            window_index = set(range(len(hs)))
            window_index = list(window_index - set(pos_list))
            hard_ext_last_hidden_states.append(hs[window_index, :])

            hard_list = [random.choice(range(i, l)) for i in pos_end_list]
            repetition_penalty_query.append(hs[hard_list, :])
        query_rep = torch.cat(ext_last_hidden_states)    # [B_q]
        repetition_penalty_query_rep = torch.cat(repetition_penalty_query)    # [B_q]
        hard_query_rep = torch.cat(hard_ext_last_hidden_states)
        query_rep = torch.cat(
            [self.s_proj(query_rep), self.e_proj(query_rep)], dim=-1        # [B_p, 2*E]
        )
        repetition_penalty_query_rep = torch.cat(
            [self.s_proj(repetition_penalty_query_rep), self.e_proj(repetition_penalty_query_rep)], dim=-1        # [B_p, 2*E]
        )
        hard_query_rep = torch.cat(
            [self.s_proj(hard_query_rep), self.e_proj(hard_query_rep)], dim=-1        # [B_p, 2*E]
        )

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']

        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        # extract the phrase representations
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]

        # hard phrase rep
        effective_num = dids_mask.sum(dim=-1)    # [B]
        hn_s_rep, hn_e_rep = [], []
        for o, m, s_pos, e_pos in zip(output, effective_num, dindex_s, dindex_e):
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(s_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_s_rep.append(o[effective_range, :])
            
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(e_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_e_rep.append(o[effective_range, :])
        hn_s_rep = torch.cat(hn_s_rep)
        hn_e_rep = torch.cat(hn_e_rep)
        
        # phrase rep consists of the in-batch negative and in-doc negative samples
        phrase_rep = torch.cat([s_rep, e_rep], dim=-1)    # [B_p, 2*E]
        hn_phrase_rep = torch.cat([hn_s_rep, hn_e_rep], dim=-1)    # [B_hn, 2*E]
        phrase_rep = torch.cat([phrase_rep, hn_phrase_rep, hard_query_rep], dim=0)    # [B_p, 2*E]

        query_rep, phrase_rep = F.normalize(query_rep, dim=-1), F.normalize(phrase_rep, dim=-1)
        
        dp = torch.matmul(query_rep, phrase_rep.t())    # [B_p, B_p]
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(doc_bsz), range(doc_bsz)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).mean()
        phrase_acc = (dp.max(dim=-1)[1].cpu() == torch.LongTensor(torch.arange(doc_bsz))).to(torch.float).mean().item()

        # repetitiaon penalty
        # repetition_penalty_query_rep, phrase_rep
        repetition_phrase_rep = torch.cat([s_rep, e_rep], dim=-1)    # [B_p, 2*E]
        repetition_phrase_rep = F.normalize(repetition_phrase_rep, dim=-1)
        repetition_penalty_query_rep = F.normalize(repetition_penalty_query_rep, dim=-1)
        assert repetition_penalty_query_rep.size() == repetition_phrase_rep.size()

        score_matrix = torch.matmul(repetition_penalty_query_rep, repetition_phrase_rep.t())
        gold_score = torch.diagonal(score_matrix).unsqueeze(1)
        difference_matrix = gold_score - score_matrix
        loss_matrix = self.args['margin'] - difference_matrix # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return phrase_loss, phrase_acc, cl_loss


class DensePhraseV2Encoder(nn.Module):

    '''single token will also be used for contrastive training'''

    def __init__(self, **args):
        super(DensePhraseV2Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'])
        self.pad = self.bert_tokenizer.pad_token_id
        self.sep = self.bert_tokenizer.sep_token_id
        self.hn_num = self.args['hard_neg_for_each_doc']
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2Model.from_pretrained(model_name)
        self.s_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)        
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)        
        )
        
        self.token_embeddings = nn.Parameter(torch.randn(len(self.tokenizer), self.model.config.hidden_size*2))

    @torch.no_grad()
    def get_phrase_rep(self, ids, ids_mask, pos, text):
        self.eval()
        rep = self.phrase_encoder(ids, ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        phrases, texts = [], []
        for rep_, pos_, text_ in zip(rep, pos, text):
            for (b, e) in pos_:
                b_rep = rep_[b, :]
                e_rep = rep_[e, :]
                p_rep = torch.cat([b_rep, e_rep], dim=-1)
                phrases.append(p_rep)
            texts.extend(text_)
        phrases = torch.stack(phrases)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        rep = self.model(input_ids=ids).last_hidden_state
        query = torch.cat([
            self.s_proj(rep[:, -1, :]),
            self.e_proj(rep[:, -1, :])
        ], dim=-1)
        query = F.normalize(query, dim=-1)
        return query

    def forward(self, batch):
        ## generation fine-tuning gpt2
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask)
        last_hidden_states = outputs.last_hidden_state

        # phrase begin position
        pos_index = batch['pos_ids']    # [B_p]
        ext_last_hidden_states = []
        hard_ext_last_hidden_states = []
        for hs, pos_list in zip(last_hidden_states, pos_index):
            # hs: [S, E]
            ext_last_hidden_states.append(hs[pos_list, :])
            # window_index = set(range(len(hs)))
            # window_index = list(window_index - set(pos_list))
            # hard_ext_last_hidden_states.append(hs[window_index, :])
        query_rep = torch.cat(ext_last_hidden_states)    # [B_q]
        # hard_query_rep = torch.cat(hard_ext_last_hidden_states)
        query_rep = torch.cat(
            [self.s_proj(query_rep), self.e_proj(query_rep)], dim=-1        # [B_p, 2*E]
        )
        # hard_query_rep = torch.cat(
        #     [self.s_proj(hard_query_rep), self.e_proj(hard_query_rep)], dim=-1        # [B_p, 2*E]
        # )

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']

        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        # extract the phrase representations
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]

        # hard phrase rep
        effective_num = dids_mask.sum(dim=-1)    # [B]
        hn_s_rep, hn_e_rep = [], []
        for o, m, s_pos, e_pos in zip(output, effective_num, dindex_s, dindex_e):
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(s_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_s_rep.append(o[effective_range, :])
            
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(e_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_e_rep.append(o[effective_range, :])
        hn_s_rep = torch.cat(hn_s_rep)
        hn_e_rep = torch.cat(hn_e_rep)
        
        # phrase rep consists of the in-batch negative and in-doc negative samples
        phrase_rep = torch.cat([s_rep, e_rep], dim=-1)    # [B_p, 2*E]
        hn_phrase_rep = torch.cat([hn_s_rep, hn_e_rep], dim=-1)    # [B_hn, 2*E]
        # phrase_rep = torch.cat([phrase_rep, hn_phrase_rep, hard_query_rep], dim=0)    # [B_p, 2*E]
        phrase_rep = torch.cat([phrase_rep, hn_phrase_rep], dim=0)    # [B_p, 2*E]

        query_rep, phrase_rep = F.normalize(query_rep, dim=-1), F.normalize(phrase_rep, dim=-1)
        
        dp = torch.matmul(query_rep, phrase_rep.t())    # [B_p, B_p]
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(doc_bsz), range(doc_bsz)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).mean()
        phrase_acc = (dp.max(dim=-1)[1].cpu() == torch.LongTensor(torch.arange(doc_bsz))).to(torch.float).mean().item()

        # token-level contrastive searching
        valid_length = ids_mask.sum(dim=-1)
        k, v = [], []
        for hidden_rep, ids_, l in zip(last_hidden_states, ids, valid_length):
            # hidden_rep: [S, E]; ids_: [S]; mask_: [S]
            label_ = ids_[1:l]
            hidden_rep_ = hidden_rep[:l-1, :]
            k.append(hidden_rep_)
            v.extend(label_)
        k = torch.cat(k)
        k = torch.cat([self.s_proj(k), self.e_proj(k)], dim=-1)
        k = F.normalize(k, dim=-1)
        dp = torch.matmul(k, F.normalize(self.token_embeddings, dim=-1).t())
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(len(k)), v] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        token_acc = (dp.max(dim=-1)[1].cpu() == torch.LongTensor(v)).to(torch.float).mean().item()

        return phrase_loss, phrase_acc, token_loss, token_acc


class DensePhraseV3Encoder(nn.Module):

    def __init__(self, **args):
        super(DensePhraseV3Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'])
        self.pad = self.bert_tokenizer.pad_token_id
        self.unk = self.bert_tokenizer.unk_token_id
        self.sep = self.bert_tokenizer.sep_token_id
        self.hn_num = self.args['hard_neg_for_each_doc']
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2Model.from_pretrained(model_name)
        self.token_embeddings = nn.Parameter(torch.randn(len(self.tokenizer), self.model.config.hidden_size*2))
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)        
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def get_phrase_rep(self, ids, ids_mask, pos, text):
        self.eval()
        rep = self.phrase_encoder(ids, ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        phrases, texts = [], []
        for rep_, pos_, text_ in zip(rep, pos, text):
            for (b, e) in pos_:
                b_rep = rep_[b, :]
                e_rep = rep_[e, :]
                p_rep = torch.cat([b_rep, e_rep], dim=-1)
                phrases.append(p_rep)
            texts.extend(text_)
        phrases = torch.stack(phrases)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        rep = self.model(input_ids=ids).last_hidden_state[:, -1, :]
        rep = torch.cat([self.s_proj(rep), self.e_proj(rep)], dim=-1)
        return rep

    def forward(self, batch):
        ## generation fine-tuning gpt2
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = torch.cat([
            self.s_proj(last_hidden_states), self.e_proj(last_hidden_states)
        ], dim=-1)

        # phrase begin position
        pos_index = batch['pos_ids']    # [B_p]
        ext_last_hidden_states = []
        hard_ext_last_hidden_states = []
        for hs, pos_list in zip(last_hidden_states, pos_index):
            ext_last_hidden_states.append(hs[pos_list, :])
        query_rep = torch.cat(ext_last_hidden_states)    # [B_q]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([s_rep, e_rep], dim=-1)    # [B_p, 2*E]
        
        # hard pharse rep v1: moving the bounding
        hard_phrase_reps_v1 = []
        valid_length = dids_mask.sum(dim=-1).tolist()
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            new_left_bounding = max(0, dindex_s_ - self.args['max_moving_step'])
            new_right_bounding = min(dindex_s_ + self.args['max_moving_step'], vl)
            indexes = list(range(new_left_bounding, new_right_bounding))
            if dindex_s_ in indexes:
                indexes.remove(dindex_s_)
            left_rep = opt[indexes]    # [B, E]
            right_rep = opt[dindex_e_].unsqueeze(0).expand(len(left_rep), -1)    # [B, E]
            rep = torch.cat([left_rep, right_rep], dim=-1)    # [B, 2*E]
            hard_phrase_reps_v1.append(rep)
        hard_phrase_reps_v1 = torch.cat(hard_phrase_reps_v1)
           
        # token_label = []
        # for ids_, pos_list, ids_ in zip(ids, pos_index, ids): 
        #     label = [ids_[i+1] for i in pos_list]
        #     token_label.extend([doc_bsz+i for i in label])

        ## 1. phrase-level contrastive learning with token negative samples
        phrase_rep = torch.cat([phrase_rep_base, hard_phrase_reps_v1], dim=0)    # [B_p, 2*E]
        dp = torch.matmul(query_rep, phrase_rep.t())
        mask = torch.zeros_like(dp)
        mask[range(doc_bsz), range(doc_bsz)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).mean()
        phrase_acc = (dp.max(dim=-1)[1].cpu() == torch.LongTensor(torch.arange(doc_bsz))).to(torch.float).mean().item()

        ## 2. token-level contrastive learning with phrase negative samples
        # phrase_rep = torch.cat([token_rep, self.token_embeddings], dim=0)    # [B_p, 2*E]
        # phrase_rep = F.normalize(phrase_rep, dim=-1)
        # dp = torch.matmul(query_rep, phrase_rep.t())    # [B_p, B_p]
        # dp /= self.args['temp']
        # remove the ground-truth token rep
        # dp[range(doc_bsz), token_label] = -1e3
        # mask = torch.zeros_like(dp)
        # mask[range(doc_bsz), range(doc_bsz)] = 1.
        # loss_ = F.log_softmax(dp, dim=-1) * mask
        # phrase_loss += (-loss_.sum(dim=1)).mean()
        # phrase_acc += (dp.max(dim=-1)[1].cpu() == torch.LongTensor(torch.arange(doc_bsz))).to(torch.float).mean().item()
        # phrase_acc /= 2

        ## 3. token-level contrastive learning
        logits = torch.matmul(last_hidden_states, self.token_embeddings.t())
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        token_loss = self.gen_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        token_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        ## 4. simctg
        # cosine_scores = torch.matmul(k, k.t()) 
        # cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        return phrase_loss, phrase_acc, token_loss, token_acc, torch.tensor(0.)

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.last_hidden_state[:, -1, :]    # [B, E]
            hidden_state = torch.cat([self.s_proj(hidden_state), self.e_proj(hidden_state)], dim=-1)
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.last_hidden_state[:, -1, :]    # [B, E]
            hidden_state = torch.cat([self.s_proj(hidden_state), self.e_proj(hidden_state)], dim=-1)
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.last_hidden_state[:, -1, :]
            hidden_state = torch.cat([self.s_proj(hidden_state), self.e_proj(hidden_state)], dim=-1)
            logit = torch.matmul(hidden_state, self.token_embeddings.t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated


class DensePhraseV4Encoder(nn.Module):

    def __init__(self, **args):
        super(DensePhraseV4Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )
        # self.token_embeddings = nn.Parameter(torch.randn(len(self.tokenizer), self.model.config.hidden_size*2))

        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep_fast(self, ids, past_key_values=None):
        self.eval()
        output = self.model(
            input_ids=ids, 
            past_key_values=past_key_values, 
            use_cache=True, 
            output_hidden_states=True
        )
        past_key_values = output['past_key_values']
        rep = F.normalize(output['hidden_states'][-1][:, -1, :], dim=-1)
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # hard negative
        '''
        hard_phrase_reps_v1_begin, hard_phrase_reps_v1_end = [], []
        valid_length = dids_mask.sum(dim=-1).tolist()
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            new_left_bounding = min(dindex_e_ + self.args['min_moving_step'], vl)
            new_right_bounding = min(dindex_e_ + self.args['max_moving_step'], vl)
            indexes = list(range(new_left_bounding, new_right_bounding))
            if indexes:
                gray_num = min(self.args['gray_cand_num'], len(indexes))
                indexes = random.sample(indexes, gray_num)
                hard_phrase_reps_v1_end.append(opt[indexes])
                hard_phrase_reps_v1_begin.extend([opt[dindex_s_]] * gray_num)
        hard_phrase_reps_v1_end = torch.cat(hard_phrase_reps_v1_end)
        hard_phrase_reps_v1_begin = torch.stack(hard_phrase_reps_v1_begin)
        assert len(hard_phrase_reps_v1_end) == len(hard_phrase_reps_v1_begin)
        hard_phrase_reps_v1 = torch.cat([self.s_proj(hard_phrase_reps_v1_begin), self.e_proj(hard_phrase_reps_v1_end)], dim=-1)
        '''

        # candidates representations
        # reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps_v1], dim=0)    # [V+B, 2*E]
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        '''
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l.item() - 1
            for i in range(l):
                phrase_query.append(hn[i])
                if i in pos_list:
                    label.append(self.vocab_size + counter)
                    counter += 1
                else:
                    label.append(ids_[i+1])
        phrase_query = torch.stack(phrase_query)
        label = torch.LongTensor(label).cuda()
        assert counter == phrase_num
        '''

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l.item() - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])

            phrase_query.append(hn[pos_list])
            phrase_label = torch.LongTensor([counter+i for i in range(len(pos_list))])
            phrase_label += self.vocab_size
            label.append(phrase_label)
            counter += len(pos_list)
        phrase_query = torch.cat(phrase_query)
        label = torch.cat(label).cuda()
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()


        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()

        ## 4. simctg
        # cosine_scores = torch.matmul(k, k.t()) 
        # cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        return loss, phrase_acc, token_acc

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated

    @torch.no_grad()
    def fast_rerank(self, ids, candidates):
        self.model.eval()
        # 1. tokenize candidates
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        # 2. prepare the ids and mask
        cids = [torch.LongTensor(t) for t in tokens]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(cids, pad_token_idx=self.pad)
        cids, mask = to_cuda(cids, mask)
        ids = ids.expand(len(cids), -1)
        seqlen = ids.size(-1)
        mask = torch.cat([torch.ones_like(ids), mask], dim=-1)
        ids = torch.cat([ids, cids], dim=-1)
        # 3. gpt2 encoding
        hidden_state = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        hidden_state = F.normalize(hidden_state, dim=-1)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, F.normalize(self.token_embeddings, dim=-1).t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)


class DensePhraseV5Encoder(nn.Module):

    '''除了phrase以外，所有的token都拿来训练，近似逼近GPT-2结果'''

    def __init__(self, **args):
        super(DensePhraseV5Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        counter = 0
        ids = ids.tolist()
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l.item() - 1
            for i in range(l):
                phrase_query.append(hn[i])
                if i in pos_list:
                    sub_label = [self.vocab_size + counter, ids_[i+1]]
                    label.append(sub_label)
                    counter += 1
                else:
                    label.append([ids_[i+1]])
        phrase_query = torch.stack(phrase_query)
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        mask = torch.zeros_like(logits)
        for i in range(len(mask)):
            mask[i, label[i]] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss = (-loss_.sum(dim=-1)).mean()

        label = torch.LongTensor([i[0] for i in label]).cuda()
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()

        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()
        return loss, phrase_acc, token_acc

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated

class DensePhraseV6Encoder(nn.Module):

    '''强负样本'''

    def __init__(self, **args):
        super(DensePhraseV6Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # hard negative moving the right boundary
        hard_phrase_reps_v1_begin, hard_phrase_reps_v1_end = [], []
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            # new_left_bounding = max(dindex_s_ - self.args['max_moving_step'], 0)
            # new_right_bounding = min(dindex_s_ + self.args['max_moving_step'], vl)
            # s_indexes = list(range(new_left_bounding, new_right_bounding))
            new_left_bounding = max(dindex_e_ - self.args['min_moving_step'], 0)
            new_right_bounding = min(dindex_e_ + self.args['max_moving_step'], vl)
            e_indexes = list(range(new_left_bounding, new_right_bounding))

            for e_ in e_indexes:
                if e_ == dindex_e_:
                    continue
                hard_phrase_reps_v1_begin.append(dindex_s_)
                hard_phrase_reps_v1_end.append(e_)
            # random_index = range(len(hard_phrase_reps_v1_end))
            # random_index = random.sample(random_index, self.args['gray_cand_num'])
            # hard_phrase_reps_v1_begin = [hard_phrase_reps_v1_begin[i] for i in random_index]
            # hard_phrase_reps_v1_end = [hard_phrase_reps_v1_end[i] for i in random_index]
            reps_begin.append(opt[hard_phrase_reps_v1_begin])
            reps_end.append(opt[hard_phrase_reps_v1_end])
        reps_begin = torch.cat(reps_begin)
        reps_end = torch.cat(reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l.item() - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])

            phrase_query.append(hn[pos_list])
            phrase_label = torch.LongTensor([counter+i for i in range(len(pos_list))])
            phrase_label += self.vocab_size
            label.append(phrase_label)
            counter += len(pos_list)
        phrase_query = torch.cat(phrase_query)
        label = torch.cat(label).cuda()
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()


        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()

        ## 4. simctg
        # cosine_scores = torch.matmul(k, k.t()) 
        # cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        return loss, phrase_acc, token_acc

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated



class DensePhraseV7Encoder(nn.Module):

    '''head, tail, and pooling'''

    def __init__(self, **args):
        super(DensePhraseV7Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )

        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 256)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 256)        
        )
        self.p_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 256)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        p_reps = []
        for rep, s, e in zip(output, dindex_s, dindex_e):
            rep_ = rep[s:e+1, :].mean(dim=0)    # [768]
            p_reps.append(rep_)
        p_rep = self.p_proj(torch.stack(p_reps))
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep), p_rep], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l.item() - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])

            phrase_query.append(hn[pos_list])
            phrase_label = torch.LongTensor([counter+i for i in range(len(pos_list))])
            phrase_label += self.vocab_size
            label.append(phrase_label)
            counter += len(pos_list)
        phrase_query = torch.cat(phrase_query)
        label = torch.cat(label).cuda()
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()


        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()

        ## 4. simctg
        # cosine_scores = torch.matmul(k, k.t()) 
        # cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        return loss, phrase_acc, token_acc

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated


class DensePhraseV8Encoder(nn.Module):

    '''add simcse loss'''

    def __init__(self, **args):
        super(DensePhraseV8Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )
        # self.token_embeddings = nn.Parameter(torch.randn(len(self.tokenizer), self.model.config.hidden_size*2))

        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)


        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l.item() - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])

            phrase_query.append(hn[pos_list])
            phrase_label = torch.LongTensor([counter+i for i in range(len(pos_list))])
            phrase_label += self.vocab_size
            label.append(phrase_label)
            counter += len(pos_list)
        phrase_query = torch.cat(phrase_query)
        label = torch.cat(label).cuda()
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()


        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()

        ## 4. simctg
        cosine_scores = torch.matmul(query, query.t()) 
        cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        loss += cl_loss
        return loss, phrase_acc, token_acc

    def compute_contrastive_loss(self, score_matrix, margin):
        difference_matrix = score_matrix - score_matrix.diagonal().unsqueeze(1)
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss 

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated

    @torch.no_grad()
    def fast_rerank(self, ids, candidates):
        self.model.eval()
        # 1. tokenize candidates
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        # 2. prepare the ids and mask
        cids = [torch.LongTensor(t) for t in tokens]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(cids, pad_token_idx=self.pad)
        cids, mask = to_cuda(cids, mask)
        ids = ids.expand(len(cids), -1)
        seqlen = ids.size(-1)
        mask = torch.cat([torch.ones_like(ids), mask], dim=-1)
        ids = torch.cat([ids, cids], dim=-1)
        # 3. gpt2 encoding
        hidden_state = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        hidden_state = F.normalize(hidden_state, dim=-1)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, F.normalize(self.token_embeddings, dim=-1).t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)

class DensePhraseV9Encoder(nn.Module):

    '''interaction layer'''

    def __init__(self, **args):
        super(DensePhraseV9Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # interaction layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args['nhead'])
        self.interaction_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['nlayer'])
        self.interaction_query_head = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Tanh(),
            nn.Linear(768, 768)
        )
        self.interaction_phrase_head = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Tanh(),
            nn.Linear(768*2, 768)
        )
        
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
        self.token_embeddings = nn.Parameter(
            list(self.model.lm_head.parameters())[0]
        )

        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)       
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)        
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
        b_phrases = self.s_proj(torch.cat(b_phrases))
        e_phrases = self.e_proj(torch.cat(e_phrases))
        phrases = torch.cat([b_phrases, e_phrases], dim=-1)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep_fast(self, ids, past_key_values=None):
        self.eval()
        output = self.model(
            input_ids=ids, 
            past_key_values=past_key_values, 
            use_cache=True, 
            output_hidden_states=True
        )
        past_key_values = output['past_key_values']
        rep = F.normalize(output['hidden_states'][-1][:, -1, :], dim=-1)
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return F.normalize(output, dim=-1)
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = ids_mask.sum(dim=-1)

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l.item() - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])

            phrase_query.append(hn[pos_list])
            phrase_label = torch.LongTensor([counter+i for i in range(len(pos_list))])
            phrase_label += self.vocab_size
            label.append(phrase_label)
            counter += len(pos_list)
        phrase_query = torch.cat(phrase_query)
        label = torch.cat(label).cuda()
        assert counter == phrase_num
        
        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        

        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()

        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label < self.vocab_size]
        token_acc = acc.to(torch.float).mean().item()

        # interaction representations
        # reps: [V+P, 768]
        cid_size, _ = logits.size()
        topk = self.args['inter_topk']
        cid_rep = query.unsqueeze(0).expand(topk + 1, -1, -1)    # [K+1, B_c, 768]
        
        logits_inter = logits.clone().detach()
        logits_inter[range(len(label)), label] = -1e3
        _, topk_index = logits_inter.topk(topk, dim=-1)    # [B_c, K]
        index = torch.cat([label.unsqueeze(1), topk_index], dim=-1)    # [B_c, K+1]
        neg_reps = torch.stack([reps[i, :] for i in index]).permute(1, 0, 2)    # [K+1, B_c, E]
        inter_reps = torch.cat([cid_rep, neg_reps], dim=-1)    # [K+1, B_c, E*2]
        inter_reps = self.interaction_phrase_head(inter_reps)    # [K+1, B_c, E]
        output = self.interaction_layer(inter_reps).permute(1, 2, 0)   #  [B_c, E*2, K+1]
        output_c = self.interaction_query_head(query).unsqueeze(1)     # [B_c, 1, E]

        logits = torch.bmm(output_c, output).squeeze(1)    # [B_c, K+1]
        mask = torch.zeros_like(logits)
        mask[:, 0] = 1.
        inter_loss_ = F.log_softmax(logits, dim=-1) * mask
        inter_loss = (-inter_loss_.sum(dim=1)).mean()
        inter_acc = (logits.max(dim=-1)[1].cpu() == torch.zeros(len(logits))).to(torch.float).mean().item()
        return loss, inter_loss, phrase_acc, token_acc, inter_acc

    @torch.no_grad()
    def nucleus_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=batch['topk'], 
                top_p=batch['topp']
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        return ''.join(self.tokenizer.convert_ids_to_tokens(generated))

    @torch.no_grad()
    def greedy_search(self, batch):
        self.eval()
        ids = batch['ids']
        generated = []
        for _ in range(batch['test_max_len']) :
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def contrastive_search(self, batch):
        self.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        generated = []
        for step in range(batch['test_max_len']):
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, -1, :]
            hidden_state = F.normalize(hidden_state)
            logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
            logit[self.unk] = -np.inf
            topk_, topk = logit.topk(batch['beam_width'], dim=-1)
            input_ids, next_index = ContrastiveDecodingOneStepGiveProb(
                self.model,
                input_ids,
                topk,
                F.softmax(topk_, dim=-1),
                batch['beam_width'],
                batch['model_prediction_confidence'],
                self.unk
            )
        input_ids = input_ids[0, prefix_length:]
        generated = ''.join(self.tokenizer.convert_ids_to_tokens(input_ids))
        return generated

    @torch.no_grad()
    def fast_rerank(self, ids, candidates):
        self.model.eval()
        # 1. tokenize candidates
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        # 2. prepare the ids and mask
        cids = [torch.LongTensor(t) for t in tokens]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(cids, pad_token_idx=self.pad)
        cids, mask = to_cuda(cids, mask)
        ids = ids.expand(len(cids), -1)
        seqlen = ids.size(-1)
        mask = torch.cat([torch.ones_like(ids), mask], dim=-1)
        ids = torch.cat([ids, cids], dim=-1)
        # 3. gpt2 encoding
        hidden_state = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        hidden_state = F.normalize(hidden_state, dim=-1)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, F.normalize(self.token_embeddings, dim=-1).t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)


