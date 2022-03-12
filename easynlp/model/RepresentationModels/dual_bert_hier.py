from model.utils import *


class BERTDualHierarchicalEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768)
        )

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])

        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            # cid_rep: L*[E]
            last_cid_rep.append(cid_rep[-1])
            # mask, [S], do not count the last utterance
            # m = torch.tensor([1] * (len(cid_rep) - 1) + [0] * (max_turn_length - len(cid_rep) + 1)).to(torch.bool)
            m = torch.tensor([1] * len(cid_rep) + [0] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        cid_mask = torch.stack(cid_mask)    # [B, S]

        # attention mechanism
        last_reps = last_reps.unsqueeze(1)    # [B, 1, E]
        attention_score = torch.bmm(last_reps, reps.permute(0, 2, 1)).squeeze(1)    # [B, S]
        attention_score /= np.sqrt(768)
        weight = torch.where(cid_mask != 0, torch.zeros_like(cid_mask), torch.ones_like(cid_mask)).cuda()    # [B, S]
        weight = weight * -1e3
        attention_score += weight
        attention_score = F.softmax(attention_score, dim=-1)

        attention_score = attention_score.unsqueeze(1)    # [B, 1, S]

        # generate the context level represetations
        # [B, 1, S] x [B, S, E]
        history_reps = torch.bmm(attention_score, reps).squeeze(1)    # [B, E]
        last_reps = last_reps.squeeze(1)
        reps = self.fusion_head(
            torch.cat([last_reps, history_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHierarchicalTrsEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.position_embedding = nn.Embedding(512, 768)
        self.args = args

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            last_cid_rep.append(cid_rep[-1])
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]

        # get the position embeddings
        bsz, seqlen, _ = reps.size()
        seqlen_index = torch.arange(seqlen).cuda().unsqueeze(0).expand(bsz, -1)    # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
        reps += pos_embd

        # 1. 
        reps = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        selected_index = torch.tensor(turn_length) - 1
        reps = reps[range(len(cid_reps)), selected_index, :]    # [B, E]
        # 2. last attention reps
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        # 3. combinatin
        reps = self.squeeze_layer(
            torch.cat([reps, last_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc

class BERTDualHierarchicalTrsMVEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsMVEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.position_embedding = nn.Embedding(512, 768)
        self.mv_num = args['mv_num']
        self.args = args

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        cid_rep = cid_rep[:, :self.mv_num, :].reshape(-1, 768)    # [B*V, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, E]
        new_turn_length = [i*self.mv_num for i in turn_length]
        cid_reps = torch.split(cid_rep, new_turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        max_turn_length_ = max(turn_length)
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            last_cid_rep.append(cid_rep[-1])
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]

        # get the position embeddings
        bsz, seqlen, _ = reps.size()
        sequence = list(chain(*[[j] * self.mv_num for j in range(max_turn_length_)]))
        seqlen_index = torch.LongTensor(sequence).cuda().unsqueeze(0).expand(bsz, -1)   # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
        reps += pos_embd

        # 1. 
        reps = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        selected_index = torch.tensor(turn_length) - 1
        reps = reps[range(len(cid_reps)), selected_index, :]    # [B, E]
        # 2. last attention reps
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        # 3. combinatin
        reps = self.squeeze_layer(
            torch.cat([reps, last_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc

class BERTDualHierarchicalTrsGPT2Encoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsGPT2Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        config = GPT2Config.from_pretrained(args['upon_model'])
        config.n_layer = 3
        self.upon_model = GPT2Model(config)
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.args = args

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            last_cid_rep.append(cid_rep[-1])
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]

        output = self.upon_model(input_ids=None, attention_mask=cid_mask, inputs_embeds=reps)
        tt = torch.LongTensor(turn_length).cuda() - 1
        reps = output.last_hidden_state[range(len(cid_reps)), tt, :]   # [B, E]
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        reps = self.squeeze_layer(
            torch.cat([reps, last_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHierarchicalTrsGPT2BothEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsGPT2BothEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        config = GPT2Config.from_pretrained(args['upon_model'])
        config.n_layer = 3
        self.upon_model = GPT2Model(config)
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*3, 768)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        self.position_embedding = nn.Embedding(512, 768)
        self.args = args

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            last_cid_rep.append(cid_rep[-1])
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]

        # get the position embeddings
        # trs reps
        trs_reps = reps.clone()
        bsz, seqlen, _ = trs_reps.size()
        seqlen_index = torch.arange(max_turn_length).cuda().unsqueeze(0).expand(bsz, -1)   # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
        trs_reps += pos_embd
        trs_reps = self.fusion_layer(
            trs_reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        selected_index = torch.tensor(turn_length) - 1
        trs_reps = reps[range(len(cid_reps)), selected_index, :]    # [B, E]

        # gpt2 reps
        output = self.upon_model(input_ids=None, attention_mask=cid_mask, inputs_embeds=reps)
        tt = torch.LongTensor(turn_length).cuda() - 1
        gpt2_reps = output.last_hidden_state[range(len(cid_reps)), tt, :]   # [B, E]
        
        # last reps
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        reps = self.squeeze_layer(
            torch.cat([trs_reps, gpt2_reps, last_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHierarchicalTrsMVColBERTEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsMVColBERTEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.position_embedding = nn.Embedding(512, 768)
        self.mv_num = args['mv_num']
        self.args = args

    def collect_cid_rep(self, cid_rep, cid_mask, turn_length):
        '''cid_rep: [B_c, S, E]; cid_mask: [B_c, S]; turn_length: [B_r]'''
        token_nums = cid_mask.sum(dim=-1).tolist()    # [B_c]
        token_nums = torch.LongTensor([min(t, self.mv_num) for t in token_nums])    # [B_c]
        turn_length_ = [item.tolist() for item in torch.split(token_nums, turn_length)]    # [B_r]
        new_turn_length = [sum(item) for item in turn_length_]    # [B_r]
        pos_index = [torch.LongTensor(list(chain(*[[idx]*t for idx, t in enumerate(item)]))) for item in turn_length_]

        reps = []
        for item, l in zip(cid_rep, token_nums):
            reps.append(item[:l, :])
        reps = torch.cat(reps, dim=0)     # [B_c*S, E]
        reps = torch.split(reps, new_turn_length)    # B_r*[S, E]
        return reps, pos_index

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        cid_rep, pos_index = self.collect_cid_rep(cid_rep, cids_mask, turn_length)    # [B*S, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B, S, E]

        # collect the rid_rep
        rid_rep = rid_rep[:, :self.mv_num, :]    # [B, S, E]
        rid_mask = rid_mask[:, :self.mv_num]
        return cid_rep, rid_rep, rid_mask, pos_index

    def get_context_level_rep(self, cid_reps, pos_index):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(item) for item in cid_reps])
        # padding by the turn_length
        reps, cid_mask, pos = [], [], []    # [B, S]
        for cid_rep, p in zip(cid_reps, pos_index):
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            p = torch.cat([p, torch.zeros(max_turn_length - len(p)).to(torch.long)], dim=-1)
            pos.append(p)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]
        pos = torch.stack(pos).cuda()    # [B, S]
        bsz, seqlen, _ = reps.size()

        # get the position embeddings
        # sequence = list(chain(*[[idx]*t for idx, t in enumerate(turn_length)]))
        # seqlen_index = torch.LongTensor(sequence).cuda().unsqueeze(0).expand(bsz, -1)   # [B, S]
        pos_embd = self.position_embedding(pos)    # [B, S, E]
        reps += pos_embd

        reps_ = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        reps = self.squeeze_layer(torch.cat([reps, reps_], dim=-1))    # [B, S, E]
        return reps, cid_mask

    def get_dot_product(self, cid_rep, rid_rep, cid_mask, rid_mask):
        # cid_rep: [B_c, S_c, E]; rid_rep: [B_r, S_r, E]
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        bsz_c, seqlen_c, _ = cid_rep.size()
        bsz_r, seqlen_r, _ = rid_rep.size()
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())

        # masking
        cid_mask = cid_mask.to(torch.long)
        cid_mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask))
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf

        #
        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        dp = dp.max(dim=-1)[0]
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep, rid_mask, pos_index = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, pos_index)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask).squeeze(0)
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep, rid_mask, pos_index = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, pos_index)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask)
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHierarchicalTrsMVColBERTMoCoEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsMVColBERTMoCoEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        self.can_encoder_momentum = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        
        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.position_embedding = nn.Embedding(512, 768)
        self.mv_num = args['mv_num']
        self.m = args['momentum_ratio']
        self.args = args

        # buffer
        self.register_buffer("queue", torch.randn(args['queue_size'], 768))    # [K, S, E]
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.can_encoder.parameters(), self.can_encoder_momentum.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys.contiguous())
        keys = F.normalize(keys, dim=-1)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size < self.args['queue_size']:
            self.queue[ptr:ptr+batch_size, :] = keys
            ptr += batch_size
        else:
            first_length = self.args['queue_size'] - ptr
            self.queue[ptr:self.args['queue_size'], :] = keys[:first_length, :]
            second_length = batch_size - first_length
            self.queue[:second_length, :] = keys[first_length:, :]
            ptr = second_length
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_ptr[0] = ptr

    def collect_cid_rep(self, cid_rep, cid_mask, turn_length):
        '''cid_rep: [B_c, S, E]; cid_mask: [B_c, S]; turn_length: [B_r]'''
        token_nums = cid_mask.sum(dim=-1).tolist()    # [B_c]
        token_nums = torch.LongTensor([min(t, self.mv_num) for t in token_nums])    # [B_c]
        turn_length_ = [item.tolist() for item in torch.split(token_nums, turn_length)]    # [B_r]
        new_turn_length = [sum(item) for item in turn_length_]    # [B_r]
        pos_index = [torch.LongTensor(list(chain(*[[idx]*t for idx, t in enumerate(item)]))) for item in turn_length_]

        reps = []
        for item, l in zip(cid_rep, token_nums):
            reps.append(item[:l, :])
        reps = torch.cat(reps, dim=0)     # [B_c*S, E]
        reps = torch.split(reps, new_turn_length)    # B_r*[S, E]
        return reps, pos_index

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        cid_rep, pos_index = self.collect_cid_rep(cid_rep, cids_mask, turn_length)    # [B*S, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, E]
        # momentum saving
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.can_encoder_momentum(rid, rid_mask)
        return cid_rep, rid_rep, pos_index, k

    def get_context_level_rep(self, cid_reps, pos_index):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(item) for item in cid_reps])
        # padding by the turn_length
        reps, cid_mask, pos = [], [], []    # [B, S]
        for cid_rep, p in zip(cid_reps, pos_index):
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            p = torch.cat([p, torch.zeros(max_turn_length - len(p)).to(torch.long)], dim=-1)
            pos.append(p)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]
        pos = torch.stack(pos).cuda()    # [B, S]
        bsz, seqlen, _ = reps.size()

        pos_embd = self.position_embedding(pos)    # [B, S, E]
        reps += pos_embd

        reps_ = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        reps = self.squeeze_layer(torch.cat([reps, reps_], dim=-1))    # [B, S, E]
        return reps, cid_mask

    def get_dot_product(self, cid_rep, rid_rep, cid_mask, test=False):
        # cid_rep: [B_c, S_c, E]; rid_rep: [B_r, E]
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1).unsqueeze(1)

        # collect the negative samples from the memory bank
        if test is False:
            rid_rep = torch.cat([rid_rep, self.queue.clone().detach().unsqueeze(1)], dim=0)    # [B_rnew, 1, E]

        bsz_c, seqlen_c, _ = cid_rep.size()
        bsz_r, seqlen_r, _ = rid_rep.size()
        rid_mask = torch.ones(bsz_r, 1).cuda()    # [B_r, 1]
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())

        # masking
        cid_mask = cid_mask.to(torch.long)
        cid_mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask))
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf

        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        # dp = dp.max(dim=-1)[0]
        dp = dp.squeeze(dim=-1)
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep, pos_index, _ = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, pos_index)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, test=True).squeeze(0)
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep, pos_index, k = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, pos_index)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, test=True)
        # dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_)
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # enqueue and dequeue
        self._dequeue_and_enqueue(k)
        return loss, acc

class BERTDualHierarchicalTrsMVGPT2Encoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsMVGPT2Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        config = GPT2Config.from_pretrained(args['upon_model'])
        config.n_layer = args['trs_nlayer']
        self.upon_model = GPT2Model(config)

        self.squeeze_layer = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768)
        )
        self.mv_num = args['mv_num']
        self.args = args

    def collect_cid_rep(self, cid_rep, cid_mask, turn_length):
        '''cid_rep: [B_c, S, E]; cid_mask: [B_c, S]; turn_length: [B_r]'''
        token_nums = cid_mask.sum(dim=-1).tolist()    # [B_c]
        token_nums = torch.LongTensor([min(t, self.mv_num) for t in token_nums])    # [B_c]
        turn_length_ = [item.tolist() for item in torch.split(token_nums, turn_length)]    # [B_r]
        new_turn_length = [sum(item) for item in turn_length_]    # [B_r]

        reps = []
        for item, l in zip(cid_rep, token_nums):
            reps.append(item[:l, :])
        reps = torch.cat(reps, dim=0)     # [B_c*S, E]
        reps = torch.split(reps, new_turn_length)    # B_r*[S, E]
        return reps

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        bsz, seqlen = rid.size()
        cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        cid_rep = self.collect_cid_rep(cid_rep, cids_mask, turn_length)    # [B*S, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B, E]
        rid_rep = rid_rep[:, :self.mv_num, :].reshape(bsz, self.mv_num*768)    # [B, M*E]
        return cid_rep, rid_rep

    def get_context_level_rep(self, cid_reps):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(item) for item in cid_reps])
        turn_length = [len(item) for item in cid_reps]
        # padding by the turn_length
        reps, cid_mask, pos = [], [], []    # [B, S]
        for cid_rep in cid_reps:
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]
        bsz, seqlen, _ = reps.size()

        output = self.upon_model(input_ids=None, attention_mask=cid_mask, inputs_embeds=reps)
        tt = torch.LongTensor(turn_length).cuda()
        reps_ = output.last_hidden_state   # [B, S, E]
        reps = self.squeeze_layer(torch.cat([reps, reps_], dim=-1))    # [B, S, E]

        reps_ = []
        for i in range(len(cid_reps)):
            ipdb.set_trace()
            rep = reps[i, tt[i]-self.mv_num:tt[i], :]    # [M, E]
            reps_.append(rep)
        reps = torch.stack(reps_).reshape(len(cid_reps), -1)    # [B, M*E]
        # reps = reps[range(len(cid_reps)), tt, :]    # [B, E]
        return reps

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(dim=0)
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps)

        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
