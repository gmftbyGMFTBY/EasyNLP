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
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
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
        rep = rep_cid + rep_rid
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


class BERTDualSCMHNGlobalEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNGlobalEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        # global vector 
        self.global_vector = nn.Parameter(torch.randn(768))
        self.args = args

    def generate_mask_matrix(self, length):
        # global vector receive the inf. from other responses, but will not influence the representations of the candidates
        mask = torch.ones(length, length).cuda()
        mask[:, 0] = 0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False, before_comp=False):
        rid_size, cid_size = len(rid), len(cid)
        # cid_rep_whole: [B_c, S, E]
        cid_rep_whole = self.ctx_encoder(cid, cid_mask, hidden=True)
        # cid_rep: [B_c, E]
        cid_rep = cid_rep_whole[:, 0, :]
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
        # rep_global: [1, B_c, E]
        rep_global = self.global_vector.unsqueeze(0).unsqueeze(0).expand(-1, cid_size, -1)
        rep_rid = torch.cat([rep_global, rep_rid], dim=0)    # [1+B_r, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [1+B_r, B_c, E]
        rep = rep_cid + rep_rid
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r, B_c, E]
        mask_matrix = self.generate_mask_matrix(len(rep))
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            tgt_mask=mask_matrix,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [B_c, E, B_r+1]
        rest = rest.permute(1, 2, 0)
        # global_vector_rep: [B_c, 1, E]
        global_vector_rep = rest[:, :, 0].unsqueeze(1)
        rest = rest[:, :, 1:]
        # dp: [B_c, B_r]
        dp = torch.bmm(global_vector_rep, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E], rep_rid: [K, B_r, E]
        rep_rid = rid_rep_whole.permute(1, 0, 2)
        rep_global = self.global_vector.unsqueeze(0).unsqueeze(0).expand(-1, cid_size, -1)
        rep_rid = torch.cat([rep_global, rep_rid], dim=0)    # [1+K, B_r, E]
        # rep_cid: [K+1, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [K+1, B_c, E]
        rep = rep_cid + rep_rid
        # rest: [K+1, B_r, E]
        mask_matrix = self.generate_mask_matrix(len(rep))
        rest = self.fusion_encoder(
            rep, 
            cid_rep_whole,
            tgt_mask=mask_matrix,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        # rest: [K+1, B_r, E] -> [B_r, E, K+1]
        rest = rest.permute(1, 2, 0)
        global_vector_rep = rest[:, :, 0].unsqueeze(1)    # [B_c, 1, E]
        rest = rest[:, :, 1:]
        # dp: [B_c, K]
        dp2 = torch.bmm(global_vector_rep, rest).squeeze(1)
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


class BERTDualSCMHN2Encoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHN2Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, is_test=False, before_comp=False):
        rid_size, cid_size = len(rid), len(cid)
        ## cid_rep_memory: [B_c, S, E]; cid_rep: [B_c, E]
        cid_rep_memory = self.ctx_encoder(cid, cid_mask, hidden=True)
        cid_rep = cid_rep_memory[:, 0, :]
        ## rid_rep: [B_r*K, E]
        rid_rep = self.can_encoder(rid, rid_mask)
        ## combine context and response embeddings before comparison
        # rep/rep_cid/rep_rid: [B_r*K, B_c, E]
        rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        rep = rep_cid + rep_rid
        ## rest: [B_r*K, B_c, E]
        rest = self.fusion_encoder(
            rep, 
            cid_rep_memory.permute(1, 0, 2),
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        ## rest: [B_c, E, B_r*K]
        rest = rest.permute(1, 2, 0)
        ## dp: [B_c, B_r*K]
        dp = torch.bmm(cid_rep.unsqueeze(1), rest).squeeze(1)
        if is_test:
            return dp
        else:
            if before_comp:
                return dp, cid_rep, rid_rep
            else:
                return dp

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
        # [B_c, B_r]
        loss = 0
        if self.args['before_comp']:
            dp, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, before_comp=True)
            # before comparsion, optimize the absolute semantic space
            # [B_c, B_r*K]
            dot_product = torch.matmul(cid_rep, rid_rep.t())
            mask = torch.zeros_like(dot_product)
            mask[range(len(cid_rep)), range(0, len(rid_rep), self.topk)] = 1.
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        else:
            dp = self._encode(cid, rid, cid_mask, rid_mask)
        mask = torch.zeros_like(dp)
        mask[range(len(cid_rep)), range(0, len(rid_rep), self.topk)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        # acc
        acc = (dp.max(dim=-1)[1] == torch.arange(0, len(rid_rep), self.topk).cuda()).to(torch.float).mean().item()
        return loss, acc

