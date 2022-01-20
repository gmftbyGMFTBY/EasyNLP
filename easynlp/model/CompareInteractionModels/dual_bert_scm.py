from model.utils import *

class BERTDualSCMSmallEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMSmallEncoder, self).__init__()
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
        self.args = args
        self.convert = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
        )

    def _encode(self, cid, rid, cid_mask, rid_mask, test=False):
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
        # cid_rep_: [B_c, 1, E]
        cid_rep_ = cid_rep_whole[:, 0, :].unsqueeze(1)
        # rid_rep: [B_r, E]
        rid_rep = self.can_encoder(rid, rid_mask)

        cid_rep_mt, rid_rep_mt = self.convert(cid_rep), self.convert(rid_rep)

        dps = []
        turn_size = len(cid) if test else self.args['small_turn_size']
        for i_b in range(0, len(cid), turn_size):
            cid_rep_p = cid_rep[i_b:i_b+turn_size]
            rid_rep_p = rid_rep[i_b:i_b+turn_size]
            cid_rep_whole_p = cid_rep_whole[i_b:i_b+turn_size, :, :]
            cid_mask_p = cid_mask[i_b:i_b+turn_size, :]
            cid_rep_p_ = cid_rep_[i_b:i_b+turn_size, :, :]
            rid_size, cid_size = len(rid_rep_p), len(cid_rep_p)
            # cid_rep: [B_r, B_c, E]
            cid_rep_p = cid_rep_p.unsqueeze(0).expand(rid_size, -1, -1)
            # rid_rep: [B_r, B_c, E]
            rid_rep_p = rid_rep_p.unsqueeze(1).expand(-1, cid_size, -1)
            # rep: [B_r, B_c, 2*E]
            rep = torch.cat([cid_rep_p, rid_rep_p], dim=-1)
            # rep: [B_r, B_c, E]
            rep = self.squeeze(rep)    

            # cid_rep_whole: [S, B_c, E]
            cid_rep_whole = cid_rep_whole_p.permute(1, 0, 2)
            # rest: [B_r, B_c, E]
            rest = self.fusion_encoder(
                rep, 
                cid_rep_whole_p,
                memory_key_padding_mask=~cid_mask_p.to(torch.bool),
            )

            ## gate
            # gate: [B_r, B_c, E]
            gate = torch.sigmoid(
                self.gate(
                    torch.cat([
                        rid_rep_p,
                        cid_rep_p,
                        rest,
                    ], dim=-1) 
                )        
            )
            # rest: [B_r, B_c, E]
            rest = gate * rid_rep_p + (1 - gate) * rest
            # rest: [B_c, E, B_r]
            rest = rest.permute(1, 2, 0)
            # dp: [B_c, B_r]
            cid_rep_p_ = F.normalize(cid_rep_p_, dim=-1)
            rest = F.normalize(rest, dim=-1)
            dp = torch.bmm(cid_rep_p_, rest).squeeze(1)
            dps.append(dp)
        return dps, cid_rep_mt, rid_rep_mt

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp, _, _ = self._encode(cid, rid, cid_mask, rid_mask)    # [1, 10]
        return dp[0].squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dps, cid_rep_mt, rid_rep_mt = self._encode(cid, rid, cid_mask, rid_mask)
        loss = 0
        # multi-task: recall training
        if self.args['coarse_recall_loss']:
            dp_mt = torch.matmul(cid_rep_mt, rid_rep_mt.t())
            mask = torch.zeros_like(dp_mt)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp_mt, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()

        # multi-task: rerank training (ranking loss)
        ## dp: [B_c, B_r]
        acc = 0
        loss_margin = 0
        for dp in dps:
            gold_score = torch.diagonal(dp).unsqueeze(dim=-1)    # [B_c, 1]
            difference = gold_score - dp    # [B_c, B_r]
            loss_matrix = torch.clamp(self.args['margin'] - difference, min=0.)   # [B_c, B_r]
            loss_margin += loss_matrix.mean()
            
            dp /= self.args['temp']
            mask = torch.zeros_like(dp)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            acc += (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        acc /= len(dps)
        return loss, loss_margin, acc



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
        self.args = args
        self.convert_res = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
        )
        self.convert_ctx = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
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

        cid_rep_mt, rid_rep_mt = self.convert_ctx(cid_rep), self.convert_res(rid_rep)

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
        cid_rep_ = F.normalize(cid_rep_, dim=-1)
        rest = F.normalize(rest, dim=-1)
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, cid_rep_mt, rid_rep_mt

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.eval()
        rest = self.can_encoder(ids, ids_mask)
        rest = self.convert_res(rest)
        return rest
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.eval()
        rest = self.ctx_encoder(ids, ids_mask)
        rest = self.convert_ctx(rest)
        return rest

    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [1, 10]

        if self.args['predict_ensemble']:
            dp_ = torch.matmul(cid_rep, rid_rep.t())
            dp = dp + dp_
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dp, cid_rep_mt, rid_rep_mt = self._encode(cid, rid, cid_mask, rid_mask)
        loss, loss_margin = 0, 0
        # multi-task: recall training
        if self.args['coarse_recall_loss']:
            dp_mt = torch.matmul(cid_rep_mt, rid_rep_mt.t())
            mask = torch.zeros_like(dp_mt)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp_mt, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()

        # multi-task: rerank training (ranking loss)
        ## dp: [B_c, B_r]
        gold_score = torch.diagonal(dp).unsqueeze(dim=-1)    # [B_c, 1]
        difference = gold_score - dp    # [B_c, B_r]
        loss_matrix = torch.clamp(self.args['margin'] - difference, min=0.)   # [B_c, B_r]
        loss_margin += loss_matrix.mean()
        
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        
        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, loss_margin, acc


class BERTDualSCMHNEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.projection = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768)
        )
        self.topk = 1 + args['gray_cand_num']
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False, before_comp=False):
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
        # rep_cid_backup: [B_r, B_c, E]
        rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        # rep = rep_cid + rep_rid
        rep = rep_rid
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
        # rid_rep_whole: [K, B_r, E], rep_rid: [K, B_r, E]
        rep_rid = rid_rep_whole.permute(1, 0, 2)
        # rep_cid: [K, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        # rep = rep_cid + rep_rid
        rep = rep_rid
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
        if before_comp:
            return dp, dp2, cid_rep, rid_rep
        else:
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
        loss = 0
        if self.args['before_comp']:
            dp, dp2, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, before_comp=True)
            cid_rep, rid_rep = self.projection(cid_rep), self.projection(rid_rep)
            # before comparsion, optimize the absolute semantic space
            dot_product = torch.matmul(cid_rep, rid_rep.t())
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        else:
            dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc

class BERTDualSCMHN2Encoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHN2Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.fusion_encoder = []
        for _ in range(args['num_layers']):
            decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
            layer = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.fusion_encoder.append(layer)
        self.fusion_encoder = nn.ModuleList(self.fusion_encoder)
        self.topk = 1 + args['gray_cand_num']
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False, before_comp=False):
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
        # rep_cid_backup: [B_r, B_c, E]
        rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        rep = rep_cid + rep_rid
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        rest = rep
        for layer_i in range(self.args['num_layers']):
            rest = self.fusion_encoder[layer_i](
                rest, 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rest += rep
        # rest: [B_c, E, B_r]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E], rep_rid: [K, B_r, E]
        rep_rid = rid_rep_whole.permute(1, 0, 2)
        # rep_cid: [K, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]
        rep = rep_cid + rep_rid
        # rest: [K, B_r, E]
        rest = rep
        for layer_i in range(self.args['num_layers']):
            rest = self.fusion_encoder[layer_i](
                rest, 
                cid_rep_whole,
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            rest += rep
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        if before_comp:
            return dp, dp2, cid_rep, rid_rep
        else:
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
        loss = 0
        if self.args['before_comp']:
            dp, dp2, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, before_comp=True)
            # before comparsion, optimize the absolute semantic space
            dot_product = torch.matmul(cid_rep, rid_rep.t())
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        else:
            dp, dp2 = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc


class BERTDualSCMRankNetEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMRankNetEncoder, self).__init__()
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
        self.args = args
        self.convert = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
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

        cid_rep_mt, rid_rep_mt = self.convert(cid_rep), self.convert(rid_rep)

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
        rest = rest.permute(1, 2, 0)    # [B_r, B_c, E]

        # dp: [B_c, B_r]
        cid_rep_ = F.normalize(cid_rep_, dim=-1)
        rest = F.normalize(rest, dim=-1)
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, cid_rep_mt, rid_rep_mt

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [1, 10]
        # dp = torch.matmul(cid_rep, rid_rep.t())
        return dp.squeeze()
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        dp, cid_rep_mt, rid_rep_mt = self._encode(cid, rid, cid_mask, rid_mask)
        loss = 0
        # multi-task: recall training
        if self.args['coarse_recall_loss']:
            dp_mt = torch.matmul(cid_rep_mt, rid_rep_mt.t())
            mask = torch.zeros_like(dp_mt)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dp_mt, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()

        # multi-task: rerank training (ranking loss)
        ## dp: [B_c, B_r]
        gold_score = torch.diagonal(dp).unsqueeze(dim=-1)    # [B_c, 1]
        difference = gold_score - dp    # [B_c, B_r]
        loss_matrix = torch.clamp(self.args['margin'] - difference, min=0.)   # [B_c, B_r]
        loss_margin = loss_matrix.mean()
        
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        
        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        
        return loss, loss_margin, acc
