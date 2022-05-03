from model.utils import *
from model.GenerationModels.utils import *

class DialogPLATOV1(nn.Module):

    def __init__(self, **args):
        super(DialogPLATOV1, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_tokens = set([self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        self.pad = self.tokenizer.pad_token_id
        self.unk = self.tokenizer.unk_token_id
        self.sep = self.tokenizer.sep_token_id
        self.mask = self.tokenizer.mask_token_id
        # add the special tokens: [EOU], [BOU]
        self.tokenizer.add_tokens(['[EOU]', '[BOU]'])
        self.eou, self.bou = self.tokenizer.convert_tokens_to_ids(['[EOU]', '[BOU]'])
        self.vocab_size = len(self.tokenizer)

        # model
        self.k = args['plato_k']
        self.model = BertModel.from_pretrained(model_name)
        self.config = self.model.config
        self.hidden_size = self.model.config.hidden_size
        self.role_embedding = nn.Embedding(2, self.hidden_size)
        self.turn_embedding = nn.Embedding(512, self.hidden_size)
        self.pos_embedding = nn.Embedding(512, self.hidden_size)
        self.k_embedding = nn.Embedding(self.k, self.hidden_size)
        self.lm_head = nn.Sequential(
            nn.Dropout(self.args['dropout']),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        self.rs_head = nn.Sequential(
            nn.Dropout(self.args['dropout']),
            nn.Linear(self.hidden_size, 2)        
        )
        self.bow_head = nn.Sequential(
            nn.Dropout(self.args['dropout']),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        self.k_head = nn.Sequential(
            nn.Dropout(self.args['dropout']),
            nn.Linear(self.hidden_size, self.k)
        )
        self.dropout = nn.Dropout(self.args['dropout'])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.model.config.layer_norm_eps)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.test_max_len = args['test_gen_max_len']
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.rs_criterion = nn.CrossEntropyLoss()
    
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
                self.pad,
                min(1., (step+1)/self.args['sep_smooth_length']),
                past_key_values,
                last_hidden_states,
                self.tokenizer,
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
        # ignore the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def generate_input_representations_generation(self, ids, role_ids, turn_ids, z_ids, seq_length, seq_res_length):
        # ids: [B, S+1]; role_ids/turn_ids: [B, S]; z_ids: [B]
        ids = ids[:, 1:]
        bsz, seqlen = ids.size()
        _, seqlen_ = role_ids.size()
        assert seqlen == seqlen_
        z_embd = self.k_embedding(z_ids).unsqueeze(1)    # [B, 1, E]
        role_embd = self.role_embedding(role_ids)    # [B, S, E]
        turn_embd = self.turn_embedding(turn_ids)    # [B, S, E]

        pos_index = []
        for s, r in zip(seq_length, seq_res_length):
            s -= 1
            pad_l = seqlen - s
            index = torch.cat([torch.arange(s-r), torch.arange(r), torch.zeros(pad_l)], dim=-1).to(torch.long)
            pos_index.append(index)
        pos_index = torch.stack(pos_index).cuda()
        pos_embd  = self.pos_embedding(pos_index)    # [B, S, E]

        token_embd = self.model.embeddings.word_embeddings(ids)    # [B, S, E]
        token_embd = torch.cat([z_embd, token_embd], dim=1)    # [B, S+1, E]
        zero_pad = torch.zeros(bsz, 1, self.hidden_size).cuda()
        role_embd = torch.cat([zero_pad, role_embd], dim=1)
        turn_embd = torch.cat([zero_pad, turn_embd], dim=1)
        pos_embd  = torch.cat([zero_pad, pos_embd], dim=1)
        embd = token_embd + pos_embd + turn_embd + role_embd
        embd = self.dropout(self.LayerNorm(embd))
        return embd

    def generate_input_representations(self, ids, role_ids, turn_ids, seq_length, seq_res_length):
        # ids: [B, S+1]; role_ids/turn_ids: [B, S]
        bsz, seqlen = ids.size()
        _, seqlen_ = role_ids.size()
        assert seqlen == seqlen_ + 1
        role_embd = self.role_embedding(role_ids)    # [B, S, E]
        turn_embd = self.turn_embedding(turn_ids)    # [B, S, E]
        pos_index = []
        for s, r in zip(seq_length, seq_res_length):
            pad_l = seqlen - s 
            index = torch.cat([torch.arange(s-1-r), torch.arange(r), torch.zeros(pad_l)], dim=-1).to(torch.long)
            pos_index.append(index)
        pos_index = torch.stack(pos_index).cuda()
        pos_embd  = self.pos_embedding(pos_index)    # [B, S, E]
        token_embd = self.model.embeddings.word_embeddings(ids)    # [B, S, E]
        zero_pad = torch.zeros(bsz, 1, self.hidden_size).cuda()
        role_embd = torch.cat([zero_pad, role_embd], dim=1)
        turn_embd = torch.cat([zero_pad, turn_embd], dim=1)
        pos_embd  = torch.cat([zero_pad, pos_embd], dim=1)
        embd = token_embd + pos_embd + turn_embd + role_embd
        embd = self.dropout(self.LayerNorm(embd))
        return embd

    def response_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=None,
        input_role_ids=None,
        input_turn_ids=None,
        input_z_ids=None,
        seq_length_=None,
        seq_res_length=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.model.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.generate_input_representations_generation(input_ids, input_role_ids, input_turn_ids, input_z_ids, seq_length_, seq_res_length)
        encoder_outputs = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def latent_act_recognition(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_role_ids=None,
        input_turn_ids=None,
        seq_length_=None,
        seq_res_length=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.model.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.generate_input_representations(input_ids, input_role_ids, input_turn_ids, seq_length_, seq_res_length)
        encoder_outputs = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def generate_bow_loss(self, hidden_state, tokens, mask):
        # hidden_state: [B, E]; tokens: [B, S]
        mask_mask = (tokens != self.mask).to(torch.bool)
        eou_mask = (tokens != self.eou).to(torch.bool)
        bou_mask = (tokens != self.bou).to(torch.bool)
        mask = mask.to(torch.bool)
        mask = mask & mask_mask & eou_mask & bou_mask
        mask = mask.to(torch.long)

        logits = F.log_softmax(self.bow_head(hidden_state), dim=-1)    # [B, V]
        target_logits = torch.gather(logits, 1, tokens)    # [B, S]
        assert target_logits.size() == tokens.size()
        target_logits = target_logits * mask
        bow_loss = (-target_logits.sum(dim=-1)).mean()    # [B]
        return bow_loss

    def forward(self, batch):
        ids = batch['ids']
        role_ids = batch['role_ids']
        turn_ids = batch['turn_ids']
        ids_mask = batch['ids_mask']
        pos_length, pos_response_length = batch['pos_length'], batch['pos_response_length']

        neg_ids = batch['neg_ids']
        neg_role_ids = batch['neg_role_ids']
        neg_turn_ids = batch['neg_turn_ids']
        neg_ids_mask = batch['neg_ids_mask']
        neg_length, neg_response_length = batch['neg_length'], batch['neg_response_length']

        bsz, _ = ids.size()

        # phrase 1
        outputs = self.latent_act_recognition(
            input_ids=ids,
            input_role_ids=role_ids,
            input_turn_ids=turn_ids,
            attention_mask=ids_mask,
            seq_length_=pos_length,
            seq_res_length=pos_response_length
        )
        hidden_state = outputs.last_hidden_state[:, 0, :]    # [B, H]
        ## add the bow loss
        bow_loss = self.generate_bow_loss(hidden_state, ids, ids_mask)

        z_prob = F.softmax(self.k_head(hidden_state), dim=-1)     # [B, K]
        z_sample = torch.multinomial(z_prob, num_samples=1).squeeze(dim=-1)    # [B]
        pos_rs_logits = self.rs_head(hidden_state)    # [B, 2]

        outputs = self.latent_act_recognition(
            input_ids=neg_ids,
            input_role_ids=neg_role_ids,
            input_turn_ids=neg_turn_ids,
            attention_mask=neg_ids_mask,
            seq_length_=neg_length,
            seq_res_length=neg_response_length
        )
        hidden_state = outputs.last_hidden_state[:, 0, :]    # [B, H]
        neg_rs_logits = self.rs_head(hidden_state)    # [B, 2]

        logits = torch.cat([pos_rs_logits, neg_rs_logits], dim=0)    # [B*2, 2]
        labels = torch.LongTensor([1] * bsz + [0] * bsz).cuda()
        random_index = list(range(bsz * 2))
        random.shuffle(random_index)
        logits = logits[random_index, :] 
        labels = labels[random_index]
        rs_loss = self.rs_criterion(logits, labels)

        # phrase 2
        labels, generation_mask = self.label_mask_generation(ids, pos_length, pos_response_length)
        output = self.response_generation(
            input_ids=ids,
            input_role_ids=role_ids,
            input_turn_ids=turn_ids,
            attention_mask=generation_mask,
            input_z_ids=z_sample,
            seq_length_=pos_length,
            seq_res_length=pos_response_length
        )
        logits = self.lm_head(output.last_hidden_state)[:, :-1, :]    # [B, S-1, V]
        mle_loss = self.criterion(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        # token_acc
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.view(-1) == labels.view(-1)).to(torch.long)
        valid_mask = (labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return mle_loss, rs_loss, bow_loss, gen_acc

    def calculate_ppl(self, input_ids, ids_mask, pos_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask, position_ids=pos_ids)
        logits = outputs.logits
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        return math.exp(mle_loss.item())

    def label_mask_generation(self, ids, pos_length, pos_response_length):
        '''
        pos_length/pos_response_length: a list of [B] size
        return the extended attention mask with shape [B, S, S]
        '''
        seqlen = ids.size(-1)
        attention_mask = []
        labels = []
        for ids_, pos_length_, pos_response_length_ in zip(ids, pos_length, pos_response_length):
            c_l = pos_length_ - pos_response_length_
            r_l = pos_response_length_
            pad_l = seqlen - c_l - r_l
            ids_ = ids_.tolist()[c_l:c_l+r_l]

            index = torch.arange(r_l)
            fill_in_mask = index[None, :] <= index[:, None]
            mask = torch.zeros(seqlen, seqlen)
            mask[:c_l, :c_l] = 1
            mask[c_l:c_l+r_l, :c_l] = 1
            mask[c_l:c_l+r_l, c_l:c_l+r_l] = fill_in_mask
            attention_mask.append(mask.to(torch.bool))

            ## label
            label = [self.pad] * c_l + ids_ + [self.pad] * pad_l 
            labels.append(torch.LongTensor(label[1:]))
        attention_mask = torch.stack(attention_mask).cuda()
        labels = torch.stack(labels).cuda()
        return labels, attention_mask
