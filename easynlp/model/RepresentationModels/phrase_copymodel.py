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
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'])
        self.pad = self.bert_tokenizer.pad_token_id
        self.unk = self.bert_tokenizer.unk_token_id
        self.sep = self.bert_tokenizer.sep_token_id
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
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
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # last_hidden_states = torch.cat([
        #     self.s_proj(last_hidden_states), self.e_proj(last_hidden_states)
        # ], dim=-1)

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
        phrase_query = []
        label = []
        vl = ids_mask.sum(dim=-1)
        counter = 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l.item()
            for i in range(l-1):
                phrase_query.append(hn[i])
                if i in pos_list:
                    label.append(self.vocab_size + counter)
                    counter += 1
                else:
                    label.append(ids_[i+1].item())
        phrase_query = torch.stack(phrase_query)
        assert counter == phrase_num
        logits = torch.matmul(phrase_query, reps.t())    # [Total, V+B]
        label = torch.LongTensor(label).cuda()

        loss = self.gen_loss_fct(logits, label)
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        ## 4. simctg
        # cosine_scores = torch.matmul(k, k.t()) 
        # cl_loss = self.compute_contrastive_loss(cosine_scores, self.args['margin'])
        return loss, acc

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
