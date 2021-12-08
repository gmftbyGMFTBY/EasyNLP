from model.utils import *

class GPT2TaCLEncoder(nn.Module):

    def __init__(self, **args):
        super(GPT2TaCLEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls])

        self.gpt2_encoder = GPT2LMIRModel(model=gpt2_model)
        self.bert_encoder = GPT2LMIRModel(model=gpt2_model)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.test_max_len = args['test_max_len']
        self.temp = args['temp']

    @torch.no_grad()
    def update_parameters(self):
        self.bert_encoder.load_state_dict(self.gpt2_encoder.state_dict())
        print(f'[!] update teacher model over')

    def _encode(self, ids, ids_mask, bert_ids, bert_ids_mask):
        gpt2_logits, gpt2_rep = self.gpt2_encoder(ids, ids_mask)
        # do not train the gpt2 encoder (teacher)
        with torch.no_grad():
            _, bert_rep = self.bert_encoder(ids, ids_mask)
        return gpt2_logits, gpt2_rep, bert_rep

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.gpt2_encoder.model(
            input_ids=ids, 
            attention_mask=ids_mask
        )
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.gpt2_encoder.model(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        bert_ids, bert_ids_mask = batch['bert_ids'], batch['bert_ids_mask']
        batch_size, length = gpt2_ids.size()

        gpt2_logits, gpt2_rep, bert_rep = self._encode(gpt2_ids, gpt2_ids_mask, bert_ids, bert_ids_mask)
        # gpt2 lm loss
        shift_logits = gpt2_logits[..., :-1, :].contiguous()
        shift_labels = gpt2_ids[..., 1:].contiguous()
        lm_loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # gpt2 token acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        token_acc = correct / num_targets

        # token-aware contrastive training loss
        # cosine similarity
        # bert_rep: [B, S, E]; gpt2_rep: [B, S, E]; gpt2_ids_mask: [B, S]
        bert_rep = F.normalize(bert_rep, dim=-1)
        gpt2_rep = F.normalize(gpt2_rep, dim=-1)
        cosine_sim = torch.bmm(gpt2_rep, bert_rep.permute(0, 2, 1))    # [B, S, S]
        cosine_sim /= self.temp

        # build the tacl loss
        mask = torch.zeros_like(cosine_sim)    # [B, S, S]
        mask[:, range(length), range(length)] = 1. 
        effective_num = 0
        # [PAD] must be ignored
        for i in range(batch_size):
            num_nonzero = gpt2_ids_mask[i].nonzero().size(0)
            mask[i][range(num_nonzero, length), range(num_nonzero, length)] = 0.
            effective_num += num_nonzero
        loss_ = F.log_softmax(cosine_sim, dim=-1) * mask    # [B, S, S]
        tacl_loss = -loss_.sum(dim=2)    # [B, S]
        tacl_loss = tacl_loss.view(-1).sum()    # [B*S]
        tacl_loss /= effective_num
        
        # tacl acc
        acc_num = 0
        for dp, mask_ in zip(cosine_sim, mask):
            # dp: [S, S]; mask_: [S, S]
            num_nonzero = mask_.nonzero().size(0)
            dp = dp[:num_nonzero, :num_nonzero]
            acc_num += (dp.max(dim=-1)[1] == torch.arange(num_nonzero).cuda()).sum().item()
        tacl_acc = acc_num / effective_num
        return lm_loss, tacl_loss, token_acc, tacl_acc


class GPT2TaCLV2Encoder(nn.Module):

    def __init__(self, **args):
        super(GPT2TaCLV2Encoder, self).__init__()
        # special tokens
        gpt2_model = args['gpt2_pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(gpt2_model)
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls])

        self.gpt2_encoder = GPT2LMIRModel(model=gpt2_model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.test_max_len = args['test_max_len']
        self.temp = args['temp']

    def _encode(self, ids, ids_mask, da_ids, da_ids_mask, da_pos_ids):
        gpt2_logits, gpt2_rep = self.gpt2_encoder(ids, ids_mask)
        da_gpt2_logits, da_gpt2_rep = self.gpt2_encoder(da_ids, da_ids_mask, pos_ids=da_pos_ids)
        # normalization
        gpt2_rep, da_gpt2_rep = F.normalize(gpt2_rep, dim=-1), F.normalize(da_gpt2_rep, dim=-1)
        return gpt2_logits, gpt2_rep, da_gpt2_logits, da_gpt2_rep

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.gpt2_encoder.model(
            input_ids=ids, 
            attention_mask=ids_mask
        )
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.gpt2_encoder.model(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def get_lm_loss_and_token_acc(self, gpt2_logits, gpt2_ids):
        # gpt2 lm loss
        shift_logits = gpt2_logits[..., :-1, :].contiguous()
        shift_labels = gpt2_ids[..., 1:].contiguous()
        lm_loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # gpt2 token acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        token_acc = correct / num_targets
        return lm_loss, token_acc

    def get_tacl_and_acc_v2(self, gpt2_rep, da_gpt2_rep, gpt2_ids_mask, da_ids_mask, da_ids, gpt2_ids):
        '''inner-sentence and inner-batch negative samples'''
        batch_size, length = gpt2_ids_mask.size()
        # da_gpt2_rep/gpt2_rep: [B*S, E]
        da_gpt2_rep = da_gpt2_rep.view(-1, da_gpt2_rep.size(-1))
        gpt2_rep = gpt2_rep.view(-1, gpt2_rep.size(-1))
        cosine_sim = torch.matmul(da_gpt2_rep, gpt2_rep.t())    # [B*S, B*S]
        cosine_sim /= self.temp
        # build the tacl loss
        mask = torch.zeros(batch_size, length, batch_size*length).cuda()    # [B, S, B*S]
        mask[:, range(length), range(length)] = 1. 
        effective_num = 0
        valid_flag = []
        # padding tokens must be ignored
        for i in range(batch_size):
            num_nonzero   = gpt2_ids_mask[i].nonzero().size(0)
            num_effective = da_ids_mask[i].nonzero().size(0)
            delta_len = num_nonzero - num_effective
            # ignore right padding tokens 
            mask[i][range(num_nonzero, length), :] = 0.
            # reset the right padding tokens similarity as -1 / temperature
            cosine_sim[
                length*i+num_nonzero:length*(i+1), 
                length*i+num_nonzero:length*(i+1)
            ] = -1. / self.temp
            # ignore left padding tokens
            mask[i][range(delta_len), :] = 0.
            effective_num += num_effective
            valid_flag.extend([0] * delta_len + [1] * num_effective + [0] * (length - num_nonzero))

        # same token must be ignored: [B*S, B*S]
        same_token_mask = da_ids.view(-1).unsqueeze(1).expand(-1, batch_size*length) ==\
            gpt2_ids.view(-1).unsqueeze(0).expand(batch_size*length, -1)
        same_token_mask[range(length), range(length)] = False
        same_token_mask = torch.cat([
            torch.cat([
                same_token_mask[1:, 1:],
                torch.BoolTensor([False] * (batch_size*length-1)).unsqueeze(0).cuda(),
            ], dim=0),    # [S, S]
            torch.BoolTensor([False] * batch_size * length).unsqueeze(1).cuda(),
        ], dim=-1)
        cosine_sim.masked_fill_(same_token_mask, -1./self.temp)

        mask = mask.view(-1, mask.size(-1))    # [B*S, B*S]
        loss_ = F.log_softmax(cosine_sim, dim=-1) * mask    # [B*S, B*S]
        tacl_loss = (-loss_.sum(dim=-1)).sum() / effective_num    # [B*S]
        
        # tacl acc
        acc_flag = cosine_sim.max(dim=-1)[1] == torch.arange(length*batch_size).cuda()    # [B*S]
        valid_flag = torch.LongTensor(valid_flag).to(torch.bool).cuda()
        tacl_acc = (acc_flag & valid_flag).to(torch.float).mean().item()
        return tacl_loss, tacl_acc
    
    def get_tacl_and_acc(self, gpt2_rep, da_gpt2_rep, gpt2_ids_mask, da_ids_mask, da_ids, gpt2_ids):
        # cosine similarity with small temperature
        # must be da_gpt2_rep multiple gpt2_rep, reverse the order is not good
        cosine_sim = torch.bmm(da_gpt2_rep, gpt2_rep.permute(0, 2, 1))    # [B, S, S]
        cosine_sim /= self.temp
        # build the tacl loss
        batch_size, length = gpt2_ids_mask.size()
        mask = torch.zeros_like(cosine_sim)    # [B, S, S]
        mask[:, range(length), range(length)] = 1. 
        effective_num = 0
        # padding tokens must be ignored
        for i in range(batch_size):
            num_nonzero   = gpt2_ids_mask[i].nonzero().size(0)
            num_effective = da_ids_mask[i].nonzero().size(0)
            delta_len = num_nonzero - num_effective
            # ignore right padding tokens 
            mask[i][range(num_nonzero, length), range(num_nonzero, length)] = 0.
            # reset the right padding tokens similarity as -1 / temperature
            cosine_sim[i][num_nonzero:, num_nonzero:] = -1./self.temp
            # ignore left padding tokens
            mask[i][range(0, delta_len), range(0, delta_len)] = 0.
            # same token must be ignored: [S, S]
            same_token_mask = da_ids[i].unsqueeze(1).expand(-1, length) ==\
                gpt2_ids[i].unsqueeze(0).expand(length, -1)
            # same token mask donot mask it-self
            same_token_mask[range(length), range(length)] = False
            same_token_mask = torch.cat([
                torch.cat([
                    same_token_mask[1:, 1:],
                    torch.BoolTensor([False] * (length-1)).unsqueeze(0).cuda(),
                ], dim=0),    # [S, S]
                torch.BoolTensor([False] * length).unsqueeze(1).cuda(),
            ], dim=-1)
            cosine_sim[i].masked_fill_(same_token_mask, -1./self.temp)
            effective_num += num_effective
        loss_ = F.log_softmax(cosine_sim, dim=-1) * mask    # [B, S, S]
        tacl_loss = -loss_.sum(dim=2)    # [B, S]
        tacl_loss = tacl_loss.view(-1).sum()    # [B*S]
        tacl_loss /= effective_num
        
        # tacl acc
        acc_num = 0
        for i in range(batch_size):
            num_nonzero   = gpt2_ids_mask[i].nonzero().size(0)
            num_effective = da_ids_mask[i].nonzero().size(0)
            dp = cosine_sim[i][num_nonzero-num_effective:num_nonzero, :]    # [S_da, S]
            acc_num += (dp.max(dim=-1)[1] == torch.arange(num_nonzero-num_effective, num_nonzero).cuda()).sum().item()
        tacl_acc = acc_num / effective_num
        return tacl_loss, tacl_acc
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        da_ids, da_ids_mask, da_pos_ids = batch['da_ids'], batch['da_ids_mask'], batch['da_pos_ids']
        batch_size, length = gpt2_ids.size()

        gpt2_logits, gpt2_rep, da_gpt2_logits, da_gpt2_rep = self._encode(
            gpt2_ids, 
            gpt2_ids_mask, 
            da_ids, 
            da_ids_mask,
            da_pos_ids,    
        )

        ## lm loss and token acc
        lm_loss_1, token_acc_1 = self.get_lm_loss_and_token_acc(gpt2_logits, gpt2_ids)
        lm_loss_2, token_acc_2 = self.get_lm_loss_and_token_acc(da_gpt2_logits, da_ids)
        lm_loss = (lm_loss_1 + lm_loss_2) / 2
        token_acc = (token_acc_1 + token_acc_2) / 2
        ## token-aware contrastive training loss
        tacl_loss, tacl_acc = self.get_tacl_and_acc(gpt2_rep, da_gpt2_rep, gpt2_ids_mask, da_ids_mask, da_ids, gpt2_ids)
        return lm_loss, tacl_loss, token_acc, tacl_acc
