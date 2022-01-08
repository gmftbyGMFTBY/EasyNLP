from model.utils import *


class BERTDualSCMHNDMEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSCMHNDMEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.topk = 1 + args['gray_cand_num']
        self.dm_turn_num = args['dm_turn_num']
        self.dm_min_compare_num = args['dm_min_compare_num']
        self.dm_max_compare_num = args['dm_max_compare_num']
        self.args = args

    def dm_turn_comparison(self, cid_rep, rid_rep, cid_rep_whole, cid_mask):
        # cid_rep: [B_c, E]; rid_rep: [B_r, K, E];
        # cid_rep_whole: [B_c, S, E]; cid_mask: [B_c, S]
        cid_size = len(cid_rep)
        rid_pool = rid_rep.view(-1, rid_rep.size(-1))    # [B_r*K, E]
        rid_pool_size = len(rid_pool)
        gt_rep_rid = rid_rep[:, 0, :].unsqueeze(0)    # [1, B_c, E]
        dps = []
        for _ in range(self.dm_turn_num):
            overall_num = random.randint(self.dm_min_compare_num, self.dm_max_compare_num)
            ## random select some responses for comparison
            ## the overall comparison number (hard + easy)
            ## rep_rid: [K_, B_c, E]; groundtruth on the position 0th
            ## collect other responses for each instance
            ngt_rep_rid = []    # [K_-1, B_c, E]
            for batch_i in range(cid_size):
                # hard negative, at least one hard negative
                hn_num = random.randint(1, min(self.topk-1, overall_num-1))
                hn_random_index = list(range(self.topk))[1:]
                random.shuffle(hn_random_index)
                hn_random_index = hn_random_index[:hn_num]
                hn_rid_rep = rid_rep[batch_i, hn_random_index, :]    # [Hn, E]
                # easy negative, at least one easy negative
                en_num = overall_num - hn_num
                # mask the correpsonding ground-truth and hard negative of batch_i instance
                en_random_index = [i for i in range(rid_pool_size) if i not in set(range(self.topk*batch_i, self.topk*batch_i+self.topk))]
                random.shuffle(en_random_index)
                en_random_index = en_random_index[:en_num]
                en_rid_rep = rid_pool[en_random_index, :]    # [En, E]
                # negative samples
                n_rep_rid = torch.cat([hn_rid_rep, en_rid_rep], dim=0)    # [K_-1, E]
                ngt_rep_rid.append(n_rep_rid)
            ngt_rep_rid = torch.stack(ngt_rep_rid).permute(1, 0, 2)    # [K_-1, B_c, E]
            rep_rid = torch.cat([gt_rep_rid, ngt_rep_rid], dim=0)    # [K, B_c, E]
            rep_cid = cid_rep.unsqueeze(0).expand(len(rep_rid), -1, -1)
            rep = rep_cid + rep_rid
            # rest: [K_, B_c, E]
            rest = self.fusion_encoder(
                rep, 
                cid_rep_whole.permute(1, 0, 2),
                memory_key_padding_mask=~cid_mask.to(torch.bool),
            )
            # rest: [B_c, E, K_]
            rest = rest.permute(1, 2, 0)
            dp = torch.bmm(cid_rep.unsqueeze(1), rest).squeeze(1)
            dps.append(dp)    # [B_c, K_]
        return dps

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

        ## dynamic comparsion turn
        dm_dp = self.dm_turn_comparison(cid_rep, rid_rep_whole, cid_rep_whole, cid_mask)

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
            return dp, dp2, dm_dp, cid_rep, rid_rep
        else:
            return dp, dp2, dm_dp

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
            dp, dp2, dp_dm, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, before_comp=True)
            # before comparsion, optimize the absolute semantic space
            dot_product = torch.matmul(cid_rep, rid_rep.t())
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1.
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        else:
            dp, dp2, dp_dm = self._encode(cid, rid, cid_mask, rid_mask)
        # fully comparison
        mask = torch.zeros_like(dp)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()
        # acc
        acc = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).to(torch.float).mean().item()
        # hard negative comparison
        # mask = torch.zeros_like(dp2)
        # mask[:, 0] = 1.
        # loss_ = F.log_softmax(dp2, dim=-1) * mask
        # loss += (-loss_.sum(dim=1)).mean()
        # dynamic comparison
        dm_loss = 0
        for dp in dp_dm:
            mask = torch.zeros_like(dp)
            mask[range(len(dp)), 0] = 1.
            loss_ = F.log_softmax(dp, dim=-1) * mask
            dm_loss += (-loss_.sum(dim=1)).mean()
        dm_loss /= len(dp_dm)
        loss += dm_loss
        return loss, acc
