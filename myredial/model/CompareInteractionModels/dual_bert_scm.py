from model.utils import *


class BERTDualSCMEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])

        # sequeeze and gate
        self.squeeze = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768*2, 768)
        )
        self.gate = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768*3, 768)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r, E]
        rid_rep = self.can_encoder(rid, rid_mask)

        ## combine context and response embeddings before comparison
        # cid_rep: [B_r, B_c, E]
        cid_rep = cid_rep.unsqueeze(0).expand(rid_size, -1, -1)
        # rid_rep: [B_r, B_c, E]
        rid_rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # rep: [B_r, B_c, 2*E]
        rep = torch.cat([cid_rep, rid_rep], dim=-1)
        # rep: [B_r, B_c, E]
        rep = self.squeeze(rep)    

        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )

        ## gate
        # gate: [B_r, B_c, E]
        gate = torch.sigmoid(
            self.gate(
                torch.cat([
                    rid_rep,
                    cid_rep,
                    rest,
                ], dim=-1) 
            )        
        )
        # rest: [B_r, B_c, E]
        rest = gate * rid_rep + (1 - gate) * rest
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMHNEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']

        self.context_before_comparison = args['context_aware_before_comparison']

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]

        ## combine context and response embeddings before comparison
        if self.context_before_comparison:
            # rep_cid_backup: [B_r, B_c, E]
            rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rid_rep: [B_r, B_c, E]
            rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E]
        if self.context_before_comparison:
            # rep_rid: [K, B_r, E]
            rep_rid = rid_rep_whole.permute(1, 0, 2)
            # rep_cid: [K, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rep: [K, B_r, E]
            rep = rid_rep_whole.permute(1, 0, 2)
        # rest: [K, B_r, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        # acc = (dp2.max(dim=-1)[1] == torch.zeros(len(dp2)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMHN2Encoder(nn.Module):

    '''bigger hidden size in fusion transformer'''

    def __init__(self, **args):
        super(BERTDualSCMHN2Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args['proj_hidden_size'], 
            nhead=args['nhead']
        )
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']

        self.context_before_comparison = args['context_aware_before_comparison']

        # projections
        self.proj1 = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, args['proj_hidden_size'])
        )
        self.proj2 = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, args['proj_hidden_size'])
        )
        self.squeeze = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(args['proj_hidden_size'], 768)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]

        ## combine context and response embeddings before comparison
        if self.context_before_comparison:
            # rep_cid_backup: [B_r, B_c, E]
            rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rid_rep: [B_r, B_c, E]
            rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = self.fusion_encoder(
            self.proj1(rep), 
            self.proj2(cid_rep_whole),
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        rest = self.squeeze(rest)
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E]
        if self.context_before_comparison:
            # rep_rid: [K, B_r, E]
            rep_rid = rid_rep_whole.permute(1, 0, 2)
            # rep_cid: [K, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rep: [K, B_r, E]
            rep = rid_rep_whole.permute(1, 0, 2)
        # rest: [K, B_r, E]
        rest = self.fusion_encoder(
            self.proj1(rep), 
            self.proj2(cid_rep_whole),
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [K, B_r, E]
        rest = self.squeeze(rest)
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        # acc = (dp2.max(dim=-1)[1] == torch.zeros(len(dp2)).cuda()).to(torch.float).mean().item()
        return loss, acc

class BERTDualSCMMCHHNEncoder(nn.Module):

    '''mult comparison head (MCH)'''

    def __init__(self, **args):
        super(BERTDualSCMMCHHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # fusion layers
        self.fusion_num = args['fusion_num']
        lns, fusions = [], []
        for _ in range(args['fusion_num']):
            decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
            fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            fusions.append(fusion_encoder)
            lns.append(nn.LayerNorm(768))
        self.fusion_encoders = nn.ModuleList(fusions)
        self.lns = nn.ModuleList(lns)

        # other parameters
        self.topk = 1 + args['gray_cand_num']
        self.context_before_comparison = args['context_aware_before_comparison']

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]

        ## combine context and response embeddings before comparison
        if self.context_before_comparison:
            # rep_cid_backup: [B_r, B_c, E]
            rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rid_rep: [B_r, B_c, E]
            rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        rep_ = rep
        for i in range(self.fusion_num):
            rep_ = self.fusion_encoders[i](
                rep_ + rep, 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rep_ = rep + rep_
            rep_ = self.lns[i](rep + rep_)
        rest = rep_
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E]
        if self.context_before_comparison:
            # rep_rid: [K, B_r, E]
            rep_rid = rid_rep_whole.permute(1, 0, 2)
            # rep_cid: [K, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rep: [K, B_r, E]
            rep = rid_rep_whole.permute(1, 0, 2)
        # rest: [K, B_r, E]
        rests = []
        rep_ = rep
        for i in range(self.fusion_num):
            rep_ = self.fusion_encoders[i](
                rep_, 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rep_ = rep + rep_
            rep_ = self.lns[i](rep + rep_)
        rest = rep_
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMHNWithEasyEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNWithEasyEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        self.context_before_comparison = args['context_aware_before_comparison']
        self.easy_num = args['easy_num']
        self.hard_better = args['activate_hard_better']

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]

        ## combine context and response embeddings before comparison
        if self.context_before_comparison:
            # rep_cid_backup: [B_r, B_c, E]
            rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rid_rep: [B_r, B_c, E]
            rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison some random easy negative are used
        # rid_rep_whole: [K, B_r, E]
        if self.context_before_comparison:
            # rep_rid: [K, B_r, E]
            rep_rid = rid_rep_whole.permute(1, 0, 2)
            # rep_rid: [M, B_r, E]
            rep_rid = torch.cat([
                rep_rid,
                self.collect_easy_random_hn(rid_rep, num=self.easy_num)
            ], dim=0)
            # rep_cid: [M, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [M, B_c, E]
            rep = rep_cid + rep_rid
        else:
            # rep: [K, B_r, E]
            rep = rid_rep_whole.permute(1, 0, 2)
            # rep: [M, B_r, E]
            rep = torch.cat([
                rep,
                self.collect_easy_random_hn(rid_rep, num=self.easy_num)
            ], dim=0)
        # rest: [M, B_r, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [M, B_r, E] -> [B_r, E, M]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, M]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2

    def collect_easy_random_hn(self, rid_rep, num=5):
        # rid_rep_whole: [B_r, E]
        size = len(rid_rep)
        easy_hn = []
        index = list(range(size))
        for i in range(size):
            while True:
                random_index = random.sample(index, num)
                if i not in random_index:
                    break
            # [M, E]
            easy_hn.append(rid_rep[random_index, :])
        easy_hn = torch.stack(easy_hn)    # [B_r, M, E]
        return easy_hn.permute(1, 0, 2)    # [M, B_r, E]

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        if self.hard_better:
            # hard negative is better than random negative
            pass

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMDistHNEncoder(nn.Module):

    '''mult comparison head (MCH) with distributed'''

    def __init__(self, **args):
        super(BERTDualSCMDistHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']

        # other parameters
        self.topk = 1 + args['gray_cand_num']
        self.context_before_comparison = args['context_aware_before_comparison']

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep_whole: [B_c*Card, S, E]
        cid_rep_whole = distributed_collect_item(cid_rep_whole)
        cid_mask = distributed_collect_item(cid_mask)
        # cid_rep: [B_c*Card, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c*Card, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        cid_size *= dist.get_world_size()
        # rid_rep: [B_r*K, E]
        if is_test:
            rid_rep = self.can_encoder(rid, rid_mask)
        else:
            rid_rep = self.can_encoder(rid, rid_mask)
            # rid_rep_whole: [B_r, K, E]
            rid_rep_whole = torch.stack(torch.split(rid_rep, self.topk))
            # rid_rep: [B_r, E]
            rid_rep = rid_rep_whole[:, 0, :]
        # rid_rep: [B_r*Card, E]
        rid_rep = distributed_collect_item(rid_rep)

        ## combine context and response embeddings before comparison
        if self.context_before_comparison:
            # rep_cid_backup: [B_r*Card, B_c, E]
            rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r*Card, B_c*Card, E]
            rep = rep_cid + rep_rid
        else:
            # rid_rep: [B_r, B_c, E]
            rep = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E]
        if self.context_before_comparison:
            rid_rep_whole = distributed_collect_item(rid_rep_whole)
            # rep_rid: [K, B_r, E]
            rep_rid = rid_rep_whole.permute(1, 0, 2)
            # rep_cid: [K, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            # rep: [B_r*Card, B_c*Card, E]
            rep = rep_cid + rep_rid
        else:
            # rep: [K, B_r, E]
            rep = rid_rep_whole.permute(1, 0, 2)
        # rest: [K, B_r, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [K, B_r, E] -> [B_r, E, K]
        if dist.get_rank() == 0:
            ipdb.set_trace()
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()


        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


