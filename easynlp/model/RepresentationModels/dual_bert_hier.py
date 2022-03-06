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
        # self.ctx_encoder = BertEmbedding(model=model)
        self.ctx_encoder = TopKBertEmbedding(model=model, m=args['mv_num'], dropout=args['dropout'])
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
        # cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        # cid_rep = cid_rep[:, :self.mv_num, :].reshape(-1, 768)    # [B*V, E]
        cid_rep = self.ctx_encoder(cids, cids_mask).permute(1, 0, 2).reshape(-1, 768)    # [B*M, E]
        rid_rep = self.can_encoder(rid, rid_mask, hidden=True)    # [B, S, E]
        new_turn_length = [i*self.mv_num for i in turn_length]
        cid_reps = torch.split(cid_rep, new_turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        max_turn_length_ = max(turn_length)
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
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

        # get the position embeddings
        bsz, seqlen, _ = reps.size()
        sequence = list(chain(*[[j] * self.mv_num for j in range(max_turn_length_)]))
        seqlen_index = torch.LongTensor(sequence).cuda().unsqueeze(0).expand(bsz, -1)   # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
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
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask).squeeze(0)
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = self.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask)
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
