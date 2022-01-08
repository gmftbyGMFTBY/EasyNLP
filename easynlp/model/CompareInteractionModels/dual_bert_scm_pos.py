from model.utils import *


class BERTDualSCMPOSHNEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMPOSHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        # fusion layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.pos_embed = nn.Embedding(args['max_compare_num'], 768)    # [512, 768]
        self.sp_embd = nn.Parameter(torch.randn(768))

        # cls head
        self.cls_head = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, args['max_compare_num'])
        )
        self.cls_criterion = nn.CrossEntropyLoss()

        self.topk = 1 + args['gray_cand_num']
        self.args = args

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
        # rep_cid_backup: [B_r, B_c, E]
        rep_rid = rid_rep.unsqueeze(1).expand(-1, cid_size, -1)
        # add the cls embedding
        rep_rid = torch.cat([
            self.sp_embd.unsqueeze(0).unsqueeze(0).expand(-1, cid_size, -1),
            rep_rid,
        ], dim=0)    # [B_r+1, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # rep: [B_r, B_c, E]; also add the position embedding
        pos_rep = self.pos_embed(torch.arange(len(rep_rid)).cuda())    # [B_r+1, E]
        pos_rep = pos_rep.unsqueeze(1).expand(-1, cid_size, -1)    # [B_r+1, B_c, E]
        rep = rep_cid + rep_rid + pos_rep
        # cid_rep_whole: [S, B_c, E]
        cid_rep_whole = cid_rep_whole.permute(1, 0, 2)
        # rest: [B_r+1, B_c, E]
        tgt_mask = self.generate_fusion_mask(len(rep))
        rest = self.fusion_encoder(
            tgt=rep, 
            memory=cid_rep_whole,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        cls_rest_easy = self.cls_head(rest[0, :, :])     # [B_c, E]
        # rest: [B_c, E, B_r]
        rest = rest[1:, :, :]    
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, B_r]
        dp = torch.bmm(cid_rep_, rest).squeeze(1)
        if is_test:
            return dp

        ### hard negative comparison
        # rid_rep_whole: [K, B_r, E]
        # rep_rid: [K, B_r, E]
        rep_rid = rid_rep_whole.permute(1, 0, 2)
        rep_rid = torch.cat([
            self.sp_embd.unsqueeze(0).unsqueeze(0).expand(-1, cid_size, -1),
            rep_rid,
        ], dim=0)    # [K+1, B_r, E]
        # rep_cid: [K+1, B_c, E]
        rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
        # in order to earse the affect of the position embedding, random select some position embeddings for hard negative comparison
        cls_rest_hard_index = []
        pos_rep = []
        for _ in range(cid_size):
            index = torch.LongTensor(random.sample(range(1, cid_size), len(rep_rid))).cuda()
            cls_rest_hard_index.append(index[0].item() - 1)
            pos_rep.append(self.pos_embed(index))    # [K+1, E]
        cls_rest_hard_index = torch.LongTensor(cls_rest_hard_index).cuda()
        pos_rep = torch.stack(pos_rep).permute(1, 0, 2)    # [K+1, B_c, E]
        rep = rep_cid + rep_rid + pos_rep
        # rest: [K+1, B_r, E]
        tgt_mask = self.generate_fusion_mask(len(rep))
        rest = self.fusion_encoder(
            tgt=rep, 
            memory=cid_rep_whole,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~cid_mask.to(torch.bool),
        )
        cls_rest_hard = self.cls_head(rest[0, :, :])    # [B_c, E]
        # rest: [K, B_r, E] -> [B_r, E, K]
        rest = rest[1:, :, :]
        rest = rest.permute(1, 2, 0)
        # dp: [B_c, K]
        dp2 = torch.bmm(cid_rep_, rest).squeeze(1)
        return dp, dp2, cid_rep, rid_rep, cls_rest_easy, cls_rest_hard, cls_rest_hard_index

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        dp = self._encode(cid, rid, cid_mask, rid_mask, is_test=True)    # [1, 10]
        return dp.squeeze()

    def generate_fusion_mask(self, size):
        mask = torch.zeros(size, size).cuda()
        mask[1:, 0] = float('-inf')
        return mask
    
    def forward(self, batch):
        cid = batch['ids']
        # rid: [B_r*K, S]
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        # [B_c, B_r]
        loss = 0

        dp, dp2, cid_rep, rid_rep, cls_rest_easy, cls_rest_hard, cls_rest_hard_index = self._encode(cid, rid, cid_mask, rid_mask)

        ## global vector classification
        ## cls_rest_easy: [B_c, B_c]; cls_rest_hard: [B_c, B_c]
        cls_rest_easy = cls_rest_easy[:, :batch_size]
        cls_rest_hard = cls_rest_hard[:, :batch_size]
        ## label for easy negative: torch.arange(B_c)
        easy_label = torch.arange(batch_size).cuda()
        loss += self.cls_criterion(cls_rest_easy, easy_label)
        ## random
        loss += self.cls_criterion(cls_rest_hard, cls_rest_hard_index)

        # before comparsion, optimize the absolute semantic space
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # after compare: dot_product
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # hard negative: dot_product
        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        return loss, acc
