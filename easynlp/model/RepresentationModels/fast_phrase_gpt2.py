from model.utils import *


class FastDensePhraseEncoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            param.requires_grad = False
            # if 'encoder.layer.11' not in name:
            #     param.requires_grad = False
        # print(f'[!] only the last BERT layer is fine-tuned')
        print(f'[!] freeze the BERT encoders')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
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
        # VL is questionable
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)

class FastDensePhraseV2Encoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseV2Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        self.w_proj = nn.Sequential(
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
        phrase_rep_base = self.w_proj(
            torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)
        )    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

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


class FastDensePhraseV3Encoder(nn.Module):

    '''token'''

    def __init__(self, **args):
        super(FastDensePhraseV3Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1) # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l - 1
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

class FastDensePhraseV4Encoder(nn.Module):

    '''hard negative'''

    def __init__(self, **args):
        super(FastDensePhraseV4Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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

        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            new_left_bounding = max(dindex_e_ - self.args['min_moving_step'], 0)
            new_right_bounding = min(dindex_e_ + self.args['max_moving_step'], vl)
            e_indexes = list(range(new_left_bounding, new_right_bounding))
            if dindex_e_ in e_indexes:
                e_indexes.remove(dindex_e_)
            hard_phrase_reps_v1_end = random.choice(e_indexes)
            hard_phrase_reps_v1_begin = dindex_s_
            reps_begin.append(opt[hard_phrase_reps_v1_begin])
            reps_end.append(opt[hard_phrase_reps_v1_end])
        reps_begin = torch.stack(reps_begin)
        reps_end = torch.stack(reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
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



class FastDensePhraseV5Encoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseV5Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        self.w_proj = nn.Sequential(
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
        phrase_rep_base = self.w_proj(torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1))    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            new_left_bounding = max(dindex_e_ - self.args['min_moving_step'], 0)
            new_right_bounding = min(dindex_e_ + self.args['max_moving_step'], vl)
            e_indexes = list(range(new_left_bounding, new_right_bounding))
            if dindex_e_ in e_indexes:
                e_indexes.remove(dindex_e_)
            hard_phrase_reps_v1_end = random.choice(e_indexes)
            hard_phrase_reps_v1_begin = dindex_s_
            reps_begin.append(opt[hard_phrase_reps_v1_begin])
            reps_end.append(opt[hard_phrase_reps_v1_end])
        reps_begin = torch.stack(reps_begin)
        reps_end = torch.stack(reps_end)
        hard_phrase_reps = self.w_proj(torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1))

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

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



class FastDensePhraseV6Encoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseV6Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//3)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//3)
        )
        self.m_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//3)
        )
        self.w_proj = nn.Sequential(
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
        # collect the mean pooling reps
        m_rep = []
        for opt, dindex_s_, dindex_e_ in zip(output, dindex_s, dindex_e):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            m_rep.append(opt[dindex_s_:dindex_e_+1, :].mean(dim=0))
        m_rep = torch.stack(m_rep)

        phrase_rep_base = self.w_proj(
            torch.cat([self.s_proj(s_rep), self.e_proj(e_rep), self.m_proj(m_rep)], dim=-1)
        )    # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

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

class FastDensePhraseV7Encoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseV7Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1) # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l.item() - 1
            for i in range(l):
                phrase_query.append(hn[i])
                if i in pos_list:
                    label.append([ids_[i+1].item(), self.vocab_size + counter])
                    counter += 1
                else:
                    label.append([ids_[i+1].item()])
        phrase_query = torch.stack(phrase_query)
        # label = torch.LongTensor(label).cuda()
        assert counter == phrase_num

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        mask = torch.zeros_like(logits)
        for idx, l in enumerate(label):
            mask[idx, l] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss = (-loss_.sum(dim=-1)).mean()

        label_ = torch.LongTensor([i[-1] for i in label]).cuda()
        acc = (logits.max(dim=-1)[1] == label_)
        acc = acc[label_ >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()

        label_ = torch.LongTensor([i[0] for i in label]).cuda()
        acc = (logits.max(dim=-1)[1] == label_)
        acc = acc[label_ < self.vocab_size]
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


class FastDensePhraseV8Encoder(nn.Module):

    '''in-doc negative'''

    def __init__(self, **args):
        super(FastDensePhraseV8Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        # for name, param in self.phrase_encoder.named_parameters():
        #     if 'encoder.layer.11' not in name and 'encoder.layer.10' not in name and 'encoder.layer.9' not in name and 'encoder.layer.8' not in name and 'encoder.layer.7' not in name and 'encoder.layer.6' not in name:
        #         param.requires_grad = False
        print(f'[!] freeze the BERT encoders')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
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
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dvl = dids_mask.sum(dim=-1)
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]
        s_output = self.s_proj(output)
        e_output = self.e_proj(output)

        # collect the phrase embeddings and the labels
        s_rep, e_rep, sp_label, ep_label = [], [], [], []
        s_counter, e_counter = 0, 0
        for dids_, s_o, e_o, dvl_, s_label_, e_label_ in zip(dids, s_output, e_output, dvl, dindex_s, dindex_e):
            dvl_ = dvl_.item()
            s_label_ = s_label_.item()
            e_label_ = e_label_.item()

            # valid hard negative mask
            s_vhn_mask = dids_ != dids_[s_label_]
            e_vhn_mask = dids_ != dids_[e_label_]
            # ignore the [CLS] and [SEP] tokens
            s_vhn_mask[0] = False
            e_vhn_mask[0] = False
            s_vhn_mask[dvl_ - 1:] = False
            e_vhn_mask[dvl_ - 1:] = False

            s_vhn = torch.cat([
                s_o[s_label_, :].unsqueeze(0),
                s_o[s_vhn_mask, :]
            ], dim=0)
            e_vhn = torch.cat([
                e_o[e_label_, :].unsqueeze(0),
                e_o[e_vhn_mask, :]
            ], dim=0)
            s_rep.append(s_vhn)
            e_rep.append(e_vhn)
            sp_label.append(s_counter + self.vocab_size)
            ep_label.append(e_counter + self.vocab_size)
            s_counter += len(s_vhn)
            e_counter += len(e_vhn)
        s_rep = torch.cat(s_rep)
        e_rep = torch.cat(e_rep)
        sp_label = torch.LongTensor(sp_label).cuda()
        ep_label = torch.LongTensor(ep_label).cuda()
        phrase_num = len(s_rep)

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        # collect token embeddings
        ## 1.1
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            token_list = list(set(range(l - 1)) - set(pos_list))
            phrase_query.append(hn[token_list, :])
            label.append(ids_[[t + 1 for t in token_list]])
        ## 1.2 phrase query embeddings
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            phrase_query.append(hn[pos_list])

        label = torch.cat(label)
        phrase_query = torch.cat(phrase_query)
        phrase_query_s = phrase_query[:, :768//2]
        phrase_query_e = phrase_query[:, 768//2:]

        # begin position learning
        s_label = torch.cat([label, sp_label], dim=0).cuda()
        reps_s = torch.cat([self.token_embeddings[:, :768//2], s_rep], dim=0)    # [V+B, 2*E]
        logits = torch.matmul(phrase_query_s, reps_s.t())    # [Total, V+B]
        loss_s = self.gen_loss_fct(logits, s_label)
        
        acc = (logits.max(dim=-1)[1] == s_label)
        acc_ = acc[s_label >= self.vocab_size]
        phrase_acc_s = acc_.to(torch.float).mean().item()

        acc_ = acc[s_label < self.vocab_size]
        token_acc_s = acc_.to(torch.float).mean().item()

        # end position learning
        e_label = torch.cat([label, ep_label], dim=0).cuda()
        reps_e = torch.cat([self.token_embeddings[:, 768//2:], e_rep], dim=0)    # [V+B, 2*E]
        logits = torch.matmul(phrase_query_e, reps_e.t())    # [Total, V+B]
        loss_e = self.gen_loss_fct(logits, e_label)

        acc = (logits.max(dim=-1)[1] == e_label)
        acc_ = acc[e_label >= self.vocab_size]
        phrase_acc_e = acc_.to(torch.float).mean().item()

        acc_ = acc[e_label < self.vocab_size]
        token_acc_e = acc_.to(torch.float).mean().item()

        # token_acc
        token_acc = (token_acc_s + token_acc_e) / 2
        phrase_acc = (phrase_acc_s + phrase_acc_e) / 2
        loss = loss_s + loss_e
        return loss, phrase_acc, token_acc

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
            hidden_state = output.hidden_states[-1][:, -1, :]
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
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
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        # VL is questionable
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)


class FastDensePhraseV9Encoder(nn.Module):

    '''token + hard negative'''

    def __init__(self, **args):
        super(FastDensePhraseV9Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            # if 'encoder.layer.11' not in name:
            #     param.requires_grad = False
            param.requires_grad = False
        print(f'[!] freeze BERT layer during training')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1) # [B_p, 2*E]
        phrase_num = len(phrase_rep_base)

        # extra hard negative samples
        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        hard_reps_begin, hard_reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            begin_index, end_index = [], []
            sample_num = min(self.args['max_hn_num'], vl-2)
            sample_index = random.sample(range(1, vl-1), sample_num)
            for i in sample_index:
                if i == dindex_s_:
                    continue
                end_range = range(i + 1, min(i + 1 + self.args['max_window_size'], vl-1))
                if len(end_range) > 0:
                    begin_index.append(i)
                    end_index.append(random.choice(end_range))
            hard_reps_begin.append(opt[begin_index])
            hard_reps_end.append(opt[end_index])
        reps_begin = torch.cat(hard_reps_begin)
        reps_end = torch.cat(hard_reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B+H, 2*E]
        # reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B+H, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            l = l - 1
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


class FastDensePhraseV10Encoder(nn.Module):

    '''token_mask'''

    def __init__(self, **args):
        super(FastDensePhraseV10Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] freeze the BERT encoders')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        token_loss, token_acc = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex = batch['doc_index']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()

        s_rep, e_rep, phrase_label, counter = [], [], [], 0
        for opt, dindex_ in zip(output, dindex):
            l, r = [], []
            index = -1
            hard_negative_sample = [idx for idx, (_, _, c) in enumerate(dindex_) if c is False]
            sample_num = min(self.args['gray_cand_num'], len(hard_negative_sample))
            hard_negative_sample = set(random.sample(hard_negative_sample, sample_num))
            for idx, (a, b, c) in enumerate(dindex_):
                if c is False and idx not in hard_negative_sample:
                    continue
                if index == -1 and c is True:
                    index = len(l)
                l.append(a)
                r.append(b)
            s_rep.append(opt[l, :])
            e_rep.append(opt[r, :])
            phrase_label.append(self.vocab_size + counter + index)
            counter += len(l)
        s_rep = torch.cat(s_rep)
        e_rep = torch.cat(e_rep)
        phrase_label = torch.LongTensor(phrase_label).cuda()
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ## 1.1 token query embeddings
        token_num = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])
            token_num += len(token_index)
        ## 1.2 phrase query embeddings
        token_label_mask = []
        phrase_num = 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            phrase_query.append(hn[pos_list])
            token_label_mask.append(ids_[[t + 1 for t in pos_list]])
            phrase_num += len(pos_list)
        token_label_mask = torch.cat(token_label_mask)
        label = torch.cat(label).cuda()
        label = torch.cat([label, phrase_label], dim=0)
        phrase_query = torch.cat(phrase_query)
        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(phrase_query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        assert token_num + phrase_num == len(logits)
        logits[range(token_num, len(logits)), token_label_mask] = -1e3
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()
        
        # add the token-level loss
        loss += token_loss
        return loss, phrase_acc, token_acc

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
        ipdb.set_trace()
        # VL is questionable
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)


class FastDensePhraseV11Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss'''

    def __init__(self, **args):
        super(FastDensePhraseV11Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        token_loss = self.gen_loss_fct(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        token_loss, token_acc = self.get_token_loss(ids, last_hidden_states)

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
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            begin_indexes = list(range(vl - self.args['left_window_size']))
            if dindex_s_ in begin_indexes:
                begin_indexes.remove(dindex_s_)
            sample_num = min(len(begin_indexes), self.args['gray_cand_num'])
            begin_indexes = random.sample(begin_indexes, sample_num)
            end_indexes = []
            for b in begin_indexes:
                range_length = random.choice(range(self.args['left_window_size'], self.args['right_window_size']))
                e = min(b+range_length, vl-1)
                end_indexes.append(e)
            reps_begin.append(opt[begin_indexes])
            reps_end.append(opt[end_indexes])
        reps_begin = torch.cat(reps_begin)
        reps_end = torch.cat(reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
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

        loss += token_loss
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



class FastDensePhraseV12Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask'''

    def __init__(self, **args):
        super(FastDensePhraseV12Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)

        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        token_loss, token_acc = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        phrase_label = torch.LongTensor([self.vocab_size + i for i in range(len(phrase_rep_base))]).cuda()
        phrase_num = len(phrase_rep_base)

        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            begin_indexes = list(range(vl - self.args['left_window_size']))
            if dindex_s_ in begin_indexes:
                begin_indexes.remove(dindex_s_)
            sample_num = min(len(begin_indexes), self.args['gray_cand_num'])
            begin_indexes = random.sample(begin_indexes, sample_num)
            end_indexes = []
            for b in begin_indexes:
                range_length = random.choice(range(self.args['left_window_size'], self.args['right_window_size']))
                e = min(b+range_length, vl-1)
                end_indexes.append(e)
            reps_begin.append(opt[begin_indexes])
            reps_end.append(opt[end_indexes])
        reps_begin = torch.cat(reps_begin)
        reps_end = torch.cat(reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # candidates representations
        reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)    # [V+B, 2*E]

        # query representations
        pos_index = batch['pos_ids']    # [B_p]
        pos_end_index = batch['pos_ids_end']
        label, phrase_query = [], []
        vl = batch['vl']

        ## 1.1 token query embeddings
        token_num = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
            for s, e in zip(pos_list, pos_list_end):
                token_index -= set(range(s, e))
            token_index = torch.LongTensor(sorted(token_index))
            phrase_query.append(hn[token_index])
            label.append(ids_[token_index+1])
            token_num += len(token_index)
        ## 1.2 phrase query embeddings
        token_label_mask = []
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            phrase_query.append(hn[pos_list])
            token_label_mask.append(ids_[[t + 1 for t in pos_list]])
        token_label_mask = torch.cat(token_label_mask)
        label = torch.cat(label)
        label = torch.cat([label, phrase_label], dim=-1)
        phrase_query = torch.cat(phrase_query)

        query = F.normalize(phrase_query, dim=-1)
        reps = F.normalize(reps, dim=-1)
        logits = torch.matmul(query, reps.t())    # [Total, V+B]
        logits /= self.args['temp']

        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        assert token_num + phrase_num == len(logits)
        logits[range(token_num, len(logits)), token_label_mask] = -1e3
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss += token_loss
        return loss, phrase_acc, token_acc

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



class FastDensePhraseV13Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV13Encoder, self).__init__()
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        # self.forward = self.forward_hhn

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

    def forward_hhn(self, batch):
        '''for dataloader: CompactHNDataset; high-quality hard negative samples'''
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex = batch['doc_index']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]

        s_rep, e_rep, hs_rep, he_rep = [], [], [], []
        for opt, dindex_ in zip(output, dindex):
            index = -1
            hard_negative_sample = [idx for idx, (_, _, c) in enumerate(dindex_) if c is False]
            sample_num = min(self.args['gray_cand_num'], len(hard_negative_sample))
            hard_negative_sample = set(random.sample(hard_negative_sample, sample_num))
            l, r = [], []
            for idx, (a, b, c) in enumerate(dindex_):
                if c is False and idx not in hard_negative_sample:
                    continue
                if index == -1 and c is True:
                    index = idx
                elif c is False:
                    l.append(a)
                    r.append(b)
            assert dindex_[index][2] is True
            s_rep.append(opt[dindex_[index][0]])
            e_rep.append(opt[dindex_[index][1]])
            hs_rep.append(opt[l, :])
            he_rep.append(opt[r, :])
        s_rep = torch.stack(s_rep)
        e_rep = torch.stack(e_rep)
        hs_rep = torch.cat(hs_rep)
        he_rep = torch.cat(he_rep)
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        hard_phrase_reps = torch.cat([self.s_proj(hs_rep), self.e_proj(he_rep)], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss = phrase_loss + token_loss
        return loss, phrase_acc, token_acc


    
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

        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_begin, reps_end = [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            begin_indexes = list(range(vl - self.args['left_window_size']))
            if dindex_s_ in begin_indexes:
                begin_indexes.remove(dindex_s_)
            sample_num = min(len(begin_indexes), self.args['gray_cand_num'])
            begin_indexes = random.sample(begin_indexes, sample_num)
            end_indexes = []
            for b in begin_indexes:
                range_length = random.choice(range(self.args['left_window_size'], self.args['right_window_size']))
                e = min(b+range_length, vl-1)
                end_indexes.append(e)
            reps_begin.append(opt[begin_indexes])
            reps_end.append(opt[end_indexes])
        reps_begin = torch.cat(reps_begin)
        reps_end = torch.cat(reps_end)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss = phrase_loss + token_loss
        return loss, phrase_acc, token_acc

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores


class FastDensePhraseV14Encoder(nn.Module):

    '''
        hard negative + fine-tune last layer + token loss + token mask:
            1. 训练token的时候，用上phrase(mask ground-truth phrase)
            2. 训练phrase的时候，用上token(mask ground-truth token)
            3. phrase表示两部分组成: head, pooling
    '''

    def __init__(self, **args):
        super(FastDensePhraseV14Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.p_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        # self.forward = self.forward_hhn

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

    def forward_hhn(self, batch):
        '''for dataloader: CompactHNDataset; high-quality hard negative samples'''
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex = batch['doc_index']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]

        s_rep, e_rep, hs_rep, he_rep = [], [], [], []
        for opt, dindex_ in zip(output, dindex):
            index = -1
            hard_negative_sample = [idx for idx, (_, _, c) in enumerate(dindex_) if c is False]
            sample_num = min(self.args['gray_cand_num'], len(hard_negative_sample))
            hard_negative_sample = set(random.sample(hard_negative_sample, sample_num))
            l, r = [], []
            for idx, (a, b, c) in enumerate(dindex_):
                if c is False and idx not in hard_negative_sample:
                    continue
                if index == -1 and c is True:
                    index = idx
                elif c is False:
                    l.append(a)
                    r.append(b)
            assert dindex_[index][2] is True
            s_rep.append(opt[dindex_[index][0]])
            e_rep.append(opt[dindex_[index][1]])
            hs_rep.append(opt[l, :])
            he_rep.append(opt[r, :])
        s_rep = torch.stack(s_rep)
        e_rep = torch.stack(e_rep)
        hs_rep = torch.cat(hs_rep)
        he_rep = torch.cat(he_rep)
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
        hard_phrase_reps = torch.cat([self.s_proj(hs_rep), self.e_proj(he_rep)], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss = phrase_loss + token_loss
        return loss, phrase_acc, token_acc

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
        # e_rep = output[range(doc_bsz), dindex_e, :]
        # collect the pooling representations (fast version)
        p_rep = []
        valid_num, select_mask = [], []
        for idx, (dindex_s_, dindex_e_) in enumerate(zip(dindex_s, dindex_e)):
            select_mask.append([0] * (dindex_s_ + 1) + [1] * (dindex_e_ - dindex_s_) + [0] * (seqlen - dindex_e_ - 1))
            valid_num.append(dindex_e_ - dindex_s_)
        select_mask = torch.LongTensor(select_mask).cuda()
        valid_num = torch.LongTensor(valid_num).cuda()
        output_p = output * select_mask.unsqueeze(-1)    # [B, S, E]
        output_p = output_p.sum(dim=1)    # [B, E]
        p_rep = output_p / valid_num.unsqueeze(1)
        # phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep), self.p_proj(p_rep)], dim=-1)    # [B_p, 2*E]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.p_proj(p_rep)], dim=-1)    # [B_p, 2*E]

        # hard negative moving the right boundary
        valid_length = dids_mask.sum(dim=-1).tolist()
        reps_pool, reps_begin, reps_end = [], [], []
        for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
            # opt: [S, E]
            dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
            begin_indexes = list(range(vl - self.args['left_window_size']))
            if dindex_s_ in begin_indexes:
                begin_indexes.remove(dindex_s_)
            sample_num = min(len(begin_indexes), self.args['gray_cand_num'])
            begin_indexes = random.sample(begin_indexes, sample_num)
            end_indexes = []
            for b in begin_indexes:
                range_length = random.choice(range(self.args['left_window_size'], self.args['right_window_size']))
                e = min(b+range_length, vl-1)
                end_indexes.append(e)
            reps_begin.append(opt[begin_indexes])
            # reps_end.append(opt[end_indexes])

            opt_p = opt.unsqueeze(0).expand(len(begin_indexes), -1, -1)    # [B, S, E]
            valid_num, select_mask = [], []
            for idx, (b, e) in enumerate(zip(begin_indexes, end_indexes)):
                select_mask.append([0] * (b + 1) + [1] * (e - b) + [0] * (seqlen - e - 1))
                valid_num.append(e - b)
            select_mask = torch.LongTensor(select_mask).cuda()
            valid_num = torch.LongTensor(valid_num).cuda()
            opt_p = opt_p * select_mask.unsqueeze(-1)
            opt_p = opt_p.sum(dim=1)
            opt_p = opt_p / valid_num.unsqueeze(1)
            reps_pool.append(opt_p)
        reps_begin = torch.cat(reps_begin)
        reps_pool = torch.cat(reps_pool)
        # reps_end = torch.cat(reps_end)
        # hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end), self.p_proj(reps_pool)], dim=-1)
        hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.p_proj(reps_pool)], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss = phrase_loss + token_loss
        return loss, phrase_acc, token_acc

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
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates):
        '''query: [1, E]'''
        self.model.eval()
        # 1. tokenize candidates
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        # 2. prepare the ids and mask
        ipdb.set_trace()
        ids = [t[0] for t in tokens]    # [B]
        embds = self.retriever.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        return F.softmax(scores, dim=-1)



class FastDensePhraseV15Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV15Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask):
        self.model.eval()
        ids, ids_mask, label = ids[:, :-1], ids_mask[:, :-1], ids[:, 1:]
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)['hidden_states'][-1]     # [B, S, E]
        output = F.normalize(output.reshape(-1, output.size(-1)), dim=-1)    # [B*S, E]
        candidates = F.normalize(self.token_embeddings, dim=-1)
        logits = torch.matmul(output, candidates.t())
        logits /= 0.01
        loss = self.gen_loss_fct(logits, label.view(-1))
        ppl = math.exp(loss.item())
        return ppl

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

        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        phrase_rep_base = output[:, 0, :]    # [B, E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        loss = phrase_loss + token_loss
        return loss, phrase_acc, token_acc

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores


class FastDensePhraseV16Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)

    seed decoder to force the phrase_encoder focus more on the phrase information itself
    '''

    def __init__(self, **args):
        super(FastDensePhraseV16Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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

        # seed weak decoder
        self.seed_decoder = WeakTrsDecoder(self.args['dropout'], len(self.bert_tokenizer), 12, 3, 1, self.bert_tokenizer.pad_token_id)

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

        ## VAE loss
        vae_ids = []
        for dids_, dindex_s_, dindex_e_ in zip(dids, dindex_s, dindex_e):
            vae_ids.append(dids_[dindex_s_:dindex_e_+1])
        vae_ids = pad_sequence(vae_ids, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)
        vae_loss, vae_acc = self.seed_decoder(
            vae_ids,
            phrase_rep_base.unsqueeze(0)    # [1, B, E]       
        )

        # hard negative moving the right boundary
        # valid_length = dids_mask.sum(dim=-1).tolist()
        # reps_begin, reps_end = [], []
        # for opt, dindex_s_, dindex_e_, vl in zip(output, dindex_s, dindex_e, valid_length):
        #     dindex_s_, dindex_e_ = dindex_s_.item(), dindex_e_.item()
        #     begin_indexes = list(range(vl - self.args['left_window_size']))
        #     if dindex_s_ in begin_indexes:
        #         begin_indexes.remove(dindex_s_)
        #     sample_num = min(len(begin_indexes), self.args['gray_cand_num'])
        #     begin_indexes = random.sample(begin_indexes, sample_num)
        #     end_indexes = []
        #     for b in begin_indexes:
        #         range_length = random.choice(range(self.args['left_window_size'], self.args['right_window_size']))
        #         e = min(b+range_length, vl-1)
        #         end_indexes.append(e)
        #     reps_begin.append(opt[begin_indexes])
        #     reps_end.append(opt[end_indexes])
        # reps_begin = torch.cat(reps_begin)
        # reps_end = torch.cat(reps_end)
        # hard_phrase_reps = torch.cat([self.s_proj(reps_begin), self.e_proj(reps_end)], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        # candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_reps], dim=0)
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        # loss = phrase_loss + token_loss + vae_loss
        return phrase_loss, token_loss, vae_loss, phrase_acc, token_acc, vae_acc

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores



class FastDensePhraseV17Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    add the pure token loss
    '''

    def __init__(self, **args):
        super(FastDensePhraseV17Encoder, self).__init__()
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        # self.forward = self.forward_hhn

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
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        # loss = phrase_loss + token_loss + pure_token_loss
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc

class FastDensePhraseV18Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    add the pure token loss
    nor normalized
    '''

    def __init__(self, **args):
        super(FastDensePhraseV18Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        # self.forward = self.forward_hhn

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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()

        # add the token loss
        # loss = phrase_loss + token_loss + pure_token_loss
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
            # hidden_state = F.normalize(hidden_state)
            # logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
            # hidden_state = F.normalize(hidden_state)
            # logits = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]    # [ V]
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
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
            # hidden_state = F.normalize(hidden_state)
            # logit = torch.matmul(hidden_state, F.normalize(self.token_embeddings, dim=-1).t())[0]     # [V]
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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # hidden_state = F.normalize(hidden_state, dim=-1)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        # shift_logits = torch.matmul(shift_hidden, F.normalize(self.token_embeddings, dim=-1).t())    # [B, S, V]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        # scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc


class FastDensePhraseV19Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    add the pure token loss
    not normalized
    gpt2 vocab encoder
    '''

    def __init__(self, **args):
        super(FastDensePhraseV19Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = GPT2Model.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        # for name, param in self.phrase_encoder.named_parameters():
        #     if 'ln_f.weight' == name or 'ln_f.bias' == name:
        #         continue
        #     if 'h.11' not in name:
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh(),
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh(),
        )

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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states)

        ## phrase encoder
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(input_ids=dids, attention_mask=dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = self.s_proj(s_rep) - self.e_proj(e_rep)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        # scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc

class FastDensePhraseV20Encoder(nn.Module):

    '''hard negative + fine-tune last layer + token loss + token mask
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    add the pure token loss
    gpt2 vocab encoder
    '''

    def __init__(self, **args):
        super(FastDensePhraseV20Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = GPT2Model.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2),
            nn.Tanh(),
        )
        self.s_proj_minus = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2),
            nn.Tanh(),
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2),
            nn.Tanh(),
        )

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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states)

        ## phrase encoder
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(input_ids=dids, attention_mask=dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        s_rep_ = self.s_proj(s_rep)
        s_rep_minus = self.s_proj_minus(s_rep)
        e_rep = self.e_proj(e_rep)
        phrase_rep_base = torch.cat([s_rep_, s_rep_minus - e_rep], dim=-1)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        query_reps = F.normalize(query_reps, dim=-1)
        candidate_reps = F.normalize(candidate_reps, dim=-1)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        logits /= self.args['temp']
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, F.normalize(embds, dim=-1).t()).squeeze(0)    # [B]
        # scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            F.normalize(hs[:, :-1, :], dim=-1),
            F.normalize(self.token_embeddings, dim=-1).t()
        )
        logits /= self.args['temp']
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc


class FastDensePhraseV21Encoder(nn.Module):

    def __init__(self, **args):
        super(FastDensePhraseV21Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        #     # param.requires_grad = False
        # print(f'[!] freeze the BERT encoders')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.output_s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.output_e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.input_s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.input_e_proj = nn.Sequential(
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
    def get_query_rep(self, input_embeds):
        self.eval()
        outputs = self.model(
            inputs_embeds=input_embeds, 
            output_hidden_states=True
        )['hidden_states'][-1]
        query = outputs[:, -1, :]    # [1, E]
        return query

    def forward(self, batch):
        ## bert phrase encoder 
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        doc_bsz = len(dids)
        # not fp16
        with autocast(enabled=False):
            output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
            s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
            e_rep = output[range(doc_bsz), dindex_e, :]
            input_phrase_rep = torch.cat([self.input_s_proj(s_rep), self.input_e_proj(e_rep)], dim=-1)    # [B_p, 2*E]
            output_phrase_rep = torch.cat([self.output_s_proj(s_rep), self.output_e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        ## gpt2 encoder
        ### 1. prepare the input embeds
        ids, ids_mask, vl = batch['ids'], batch['ids_mask'], batch['vl']
        token_index = ids < len(self.tokenizer)
        phrase_index = ids >= len(self.tokenizer)

        token_ids = ids.clone()
        token_ids[phrase_index] = self.tokenizer.eos_token_id
        token_embeddings = self.model.transformer.wte(token_ids)    # [B, S, E]

        assert len(input_phrase_rep) == phrase_index.sum().item()
        phrase_index = phrase_index.unsqueeze(-1).expand(-1, -1, 768)
        token_embeddings[phrase_index] = input_phrase_rep.reshape(-1)    # [B, S, E]

        outputs = self.model(
            inputs_embeds=token_embeddings, 
            attention_mask=ids_mask,
            output_hidden_states=True
        )
        last_hidden_states = outputs.hidden_states[-1]    # [B, S, E]
        query = []
        for hs, l in zip(last_hidden_states, vl):
            query.append(hs[:l-1, :])
        query = torch.cat(query)

        ### 2. prepare the output embeddings
        output_embeds = self.token_embeddings[token_ids, :]    # [B, S, E]
        output_embeds[phrase_index] = output_phrase_rep.reshape(-1)
        candidates = []
        labels = []
        for hs, l, ti in zip(output_embeds, vl, ids):
            candidates.append(hs[1:l, :])
            labels.append(ti[1:l])
        candidates = torch.cat(candidates)
        labels = torch.cat(labels)

        ### 3. calculate the loss and accuracy
        logits = torch.matmul(query, candidates.t())
        assert len(query) == len(candidates)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), range(len(logits))] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # token acc
        token_pos = labels < len(self.tokenizer)
        phrase_pos = labels >= len(self.tokenizer)
        acc = logits.max(dim=-1)[1] == torch.arange(len(logits)).cuda()
        token_acc = acc[token_pos].to(torch.float).mean().item()
        phrase_acc = acc[phrase_pos].to(torch.float).mean().item()
        return loss, token_acc, phrase_acc

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
        # VL is questionable
        vl = mask.sum(dim=-1) - seqlen
        # confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence, dim=-1)


class FastDensePhraseV22Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV22Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        self.phrase_encoder.resize_token_embeddings(self.phrase_encoder.config.vocab_size+1)
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        return phrases, texts

    @torch.no_grad()
    def get_query_rep_beam(self, ids):
        '''batch inference with the left padding'''
        self.eval()
        max_length = max([len(i) for i in ids])
        ids = [torch.cat((torch.LongTensor([self.pad] * (max_length - len(i))), i)) for i in ids]
        ids = torch.stack(ids)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids_pos = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
        ids, ids_mask, ids_pos = to_cuda(ids, ids_mask, ids_pos)
        output = self.model(input_ids=ids, attention_mask=ids_mask, position_ids=ids_pos, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        ipdb.set_trace()
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )     # []
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        ids_mask = ids_mask[:, :-1].reshape(-1)
        ids_mask_label = ids_mask == 1
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.

        logits = logits[ids_mask_label, :]
        mask = mask[ids_mask_label, :]
        label = label[ids_mask_label]

        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc

class FastDensePhraseV23Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    with the document
    '''

    def __init__(self, **args):
        super(FastDensePhraseV23Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 320)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 320)
        )
        self.d_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 128)
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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep
    
    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        d_rep = output[range(doc_bsz), 0, :]
        phrase_rep_base = torch.cat([self.d_proj(d_rep), self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc


class FastDensePhraseV24Encoder(nn.Module):

    '''fine-tune last layer + token loss'''

    def __init__(self, **args):
        super(FastDensePhraseV24Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def get_token_loss(self, ids, hs):
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        token_loss = self.gen_loss_fct(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc
    
    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        token_loss, token_acc = self.get_token_loss(ids, last_hidden_states)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        with torch.no_grad():
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
        vl = batch['vl']

        ids = ids.cpu()
        counter = 0
        for ids_, hn, pos_list, pos_list_end, l in zip(ids, last_hidden_states, pos_index, pos_end_index, vl):
            token_index = set(range(l - 1))
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

        logits = torch.matmul(phrase_query, reps.t())    # [Total, V+B]
        loss = self.gen_loss_fct(logits, label)
        total_acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        
        acc = (logits.max(dim=-1)[1] == label)
        acc = acc[label >= self.vocab_size]
        phrase_acc = acc.to(torch.float).mean().item()
        return loss, token_loss, torch.tensor(0.), phrase_acc, token_acc, 0.

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
        return confidence
 


class FastDensePhraseV25Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV25Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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

        # config = GPT2Config.from_pretrained(self.args['pretrained_model'])
        # config.vocab_size = self.vocab_size
        # self.model = GPT2LMHeadModel(config)

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//3)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//3)
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
        rep = output['hidden_states'][-1][:, -1, :]
        return past_key_values, rep

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc, 0

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
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

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
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores

    def get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )     # []
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        ids_mask = ids_mask[:, :-1].reshape(-1)
        ids_mask_label = ids_mask == 1
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.

        logits = logits[ids_mask_label, :]
        mask = mask[ids_mask_label, :]
        label = label[ids_mask_label]

        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc



class FastDensePhraseV26Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV26Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.criterion = nn.MarginRankingLoss(margin=self.args['hard_margin'])

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
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        ## triplet loss with hardest sample in batch
        pos_reps = token_logits[range(len(token_logits)), token_labels]    # [B]
        _, topk_ids = token_logits.topk(2, dim=-1)    # [B, 2]
        neg_ids = []
        for topk_ids_, label in zip(topk_ids.tolist(), token_labels.tolist()):
            if label in topk_ids_:
                topk_ids_.remove(label)
            neg_ids.append(topk_ids_[0])
        neg_reps = token_logits[range(len(token_logits)), neg_ids]
        token_loss_triplet = self.criterion(pos_reps, neg_reps, torch.ones_like(pos_reps))

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        ## triplet loss with hardest sample in batch
        pos_reps = phrase_logits[phrase_pos_index, phrase_labels[phrase_pos_index]]    # [B']
        _, topk_ids = phrase_logits[phrase_pos_index].topk(2, dim=-1)    # [B', 2]
        neg_ids = []
        for topk_ids_, label in zip(topk_ids.tolist(), phrase_labels[phrase_pos_index]):
            if label in topk_ids_:
                topk_ids_.remove(label)
            neg_ids.append(topk_ids_[0])
        neg_reps = phrase_logits[phrase_pos_index, neg_ids]
        phrase_loss_triplet = self.criterion(pos_reps, neg_reps, torch.ones_like(pos_reps))

        # accuracy record
        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()
        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_loss_triplet, token_loss_triplet, phrase_acc, token_acc

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
            hidden_state = output.hidden_states[-1][:, -1, :]
            logits = torch.matmul(hidden_state, self.token_embeddings.t())[0]    # [ V]
            logits[self.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            generated.append(next_token.item())
            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=-1)
        if self.args['lang'] == 'zh':
            return ''.join(self.tokenizer.convert_ids_to_tokens(generated))
        else:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        hidden_state = self.h_proj(hidden_state)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)



class FastDensePhraseV27Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)

    BOW loss is used to force the model to understand the main content of the phrase
    '''

    def __init__(self, **args):
        super(FastDensePhraseV27Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.bow_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size*2, len(self.bert_tokenizer))
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
        self.criterion = nn.MarginRankingLoss(margin=self.args['hard_margin'])

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
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        ### collect the phrases
        phrase_ids = []
        for doc_id, s_idx, e_idx in zip(dids, dindex_s, dindex_e):
            phrase_ids_ = doc_id[s_idx:e_idx+1]
            phrase_ids.append(phrase_ids_)
        phrase_ids = pad_sequence(phrase_ids, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)

        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]

        # fast collect hard negative
        '''
        vl_doc = dids_mask.sum(dim=-1)
        hard_s_rep, hard_e_rep = [], []
        for vl, s, e, hd in zip(vl_doc, dindex_s, dindex_e, output):
            indexes = list(range(1, vl-1))
            try:
                indexes.remove(s)
                indexes.remove(e)
            except:
                pass
            # all the tokens are too large and will raise the OOM
            if len(indexes) > 64:
                indexes = random.sample(indexes, 64)
            random_length = np.random.randint(0, 8, len(indexes))
            hard_s_rep.append(hd[indexes, :])
            hard_e_rep.append(hd[random_length, :])
        hard_s_rep = torch.cat(hard_s_rep)
        hard_e_rep = torch.cat(hard_e_rep)
        hard_phrase_rep_base = torch.cat([self.s_proj(hard_s_rep), self.e_proj(hard_e_rep)], dim=-1)
        '''
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # get the bow loss
        # bow_loss = self.get_bow_loss(phrase_rep_base, phrase_ids)

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        # candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base, hard_phrase_rep_base], dim=0)
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        ## triplet loss with hardest sample in batch
        '''
        pos_reps = token_logits[range(len(token_logits)), token_labels]    # [B]
        _, topk_ids = token_logits.topk(2, dim=-1)    # [B, 2]
        neg_ids = []
        for topk_ids_, label in zip(topk_ids.tolist(), token_labels.tolist()):
            if label in topk_ids_:
                topk_ids_.remove(label)
            neg_ids.append(topk_ids_[0])
        neg_reps = token_logits[range(len(token_logits)), neg_ids]
        token_loss_triplet = self.criterion(pos_reps, neg_reps, torch.ones_like(pos_reps))
        '''

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        ## triplet loss with hardest sample in batch
        '''
        pos_reps = phrase_logits[phrase_pos_index, phrase_labels[phrase_pos_index]]    # [B']
        _, topk_ids = phrase_logits[phrase_pos_index].topk(2, dim=-1)    # [B', 2]
        neg_ids = []
        for topk_ids_, label in zip(topk_ids.tolist(), phrase_labels[phrase_pos_index]):
            if label in topk_ids_:
                topk_ids_.remove(label)
            neg_ids.append(topk_ids_[0])
        neg_reps = phrase_logits[phrase_pos_index, neg_ids]
        phrase_loss_triplet = self.criterion(pos_reps, neg_reps, torch.ones_like(pos_reps))
        '''
        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        # return phrase_loss, bow_loss, token_loss, pure_token_loss, phrase_loss_triplet, token_loss_triplet, phrase_acc, token_acc
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc

    def _get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        logits = logits.reshape(-1, logits.size(-1))
        label = label.reshape(-1)
        ids_mask = ids_mask[:, :-1].reshape(-1)
        ids_mask_label = ids_mask == 1
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), label] = 1.

        logits = logits[ids_mask_label, :]
        mask = mask[ids_mask_label, :]
        label = label[ids_mask_label]

        loss_ = F.log_softmax(logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (logits.max(dim=-1)[1] == label)
        token_acc = acc.to(torch.float).mean().item()
        return token_loss, token_acc

    def get_bow_loss(self, phrase_reps, phrase_ids):
        # phrase_reps: [B, H]; phrase_ids: [B, S]
        ## token mask to ignore the special tokens
        mask_unk = (phrase_ids != self.bert_tokenizer.unk_token_id)
        mask_pad = (phrase_ids != self.bert_tokenizer.pad_token_id)
        mask = mask_pad & mask_unk
        mask = mask.to(torch.long)

        logits = F.log_softmax(self.bow_head(phrase_reps), dim=-1)    # [B, V]
        target_logits = torch.gather(logits, 1, phrase_ids)    # [B, S]
        assert target_logits.size() == phrase_ids.size()
        target_logits = target_logits * mask    # [B, S]
        bow_loss = - (target_logits.sum(dim=-1)).mean()
        return bow_loss

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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        hidden_state = self.h_proj(hidden_state)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

    @torch.no_grad()
    def fast_rerank_v2(self, query, candidates, temp=1.0):
        '''query: [1, E]'''
        self.model.eval()
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        ids = [t[0] for t in tokens]    # [B]
        embds = self.token_embeddings[ids, :]     # [B, 768]
        scores = torch.matmul(query, embds.t()).squeeze(0)    # [B]
        scores = F.softmax(scores/temp, dim=-1)
        return scores








class FastDensePhraseV28Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV28Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc

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



class FastDensePhraseV29Encoder(nn.Module):

    '''fine-tune last layer + token loss + token mask + dot production
    训练token的时候，用上phrase(mask ground-truth phrase)
    训练phrase的时候，用上token(mask ground-truth token)
    '''

    def __init__(self, **args):
        super(FastDensePhraseV29Encoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'][self.args['lang']])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'][self.args['lang']])
        
        # only fine-tune the last transformer layer parameters
        for name, param in self.phrase_encoder.named_parameters():
            if 'encoder.layer.11' not in name:
                param.requires_grad = False
        print(f'[!] only the last BERT layer is fine-tuned')
        
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
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        self.h_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
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
        pure_token_loss, _ = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]
        phrase_rep_base = torch.cat([self.s_proj(s_rep), self.e_proj(e_rep)], dim=-1)    # [B_p, 2*E]

        # query representations
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
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.LongTensor(phrase_labels).cuda()
        candidate_reps = torch.cat([self.token_embeddings, phrase_rep_base], dim=0)
        logits = torch.matmul(query_reps, candidate_reps.t())    # [Total, V+B]
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # learning token, mask phrase
        token_logits = logits.clone()
        mask = torch.zeros_like(token_logits)
        mask[range(len(token_logits)), token_labels] = 1.
        token_logits[phrase_pos_index, phrase_labels[phrase_pos_index]] = -1e3
        loss_ = F.log_softmax(token_logits, dim=-1) * mask
        token_loss = (-loss_.sum(dim=1)).mean()

        acc = token_logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        token_acc = acc.to(torch.float).mean().item()

        # learning phrase, mask token
        phrase_logits = logits.clone()
        mask = torch.zeros_like(phrase_logits)
        mask[phrase_pos_index, phrase_labels[phrase_pos_index]] = 1.
        phrase_logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e3
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(phrase_logits, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).sum() / valid_num

        acc = phrase_logits[phrase_pos_index].max(dim=-1)[1] == phrase_labels[phrase_pos_index]
        phrase_acc = acc.to(torch.float).mean().item()
        return phrase_loss, token_loss, pure_token_loss, phrase_acc, token_acc

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

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
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
        hidden_state = self.h_proj(hidden_state)
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)




