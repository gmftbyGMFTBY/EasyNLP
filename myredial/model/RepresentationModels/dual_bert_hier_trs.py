from model.utils import *


class BERTDualHierarchicalTrsEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalTrsEncoder, self).__init__()
        model = args['pretrained_model']
        nalyer = args['nlayer']
        nhead = args['nhead']
        nhide = args['nhide']
        dropout = args['dropout']

        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            768,
            nhead=nhead,
            dim_feedforward=nhide,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(768)
        self.position_embd = nn.Embedding(512, 768)
        self.speaker_embd = nn.Embedding(2, 768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            nlayer,
            encoder_norm,
        )

    def _encode(self, cids, rid, cids_mask, rid_mask, recover_mapping):
        '''resort'''
        cid_reps = []
        for cid, cid_mask in zip(cids, cids_mask):
            cid_rep = self.ctx_encoder(cid, cid_mask)
            cid_reps.append(cid_rep)
        cid_reps = torch.cat(cid_reps)    # [B, E]
        # recover
        cid_reps = [cid_reps[recover_mapping[idx]] for idx in range(len(cid_reps))]
        cid_rep = torch.stack(cid_reps)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def reconstruct_tensor(self, cid_rep, cid_turn_length):
        '''resort and generate the order, context length mask'''
        # =========== reconstruct cid ========== #
        cid_rep = torch.split(cid_rep, cid_turn_length)
        # =========== padding =========== #
        max_turn_length = max([len(i) for i in cid_rep])
        cid_reps = []    # [B, S, E]
        cid_mask = []    # [B, S]
        for ctx in cid_rep:
            # mask, [S]
            m = torch.tensor([0] * len(ctx) + [1] * (max_turn_length - len(ctx))).to(torch.bool)
            cid_mask.append(m)
            if len(ctx) < max_turn_length:
                # support apex
                zero_tensor = torch.zeros(1, 768).half().cuda()
                padding = [zero_tensor] * (max_turn_length - len(ctx))
                ctx = torch.cat([ctx] + padding)    # append [S, E]
            cid_reps.append(ctx)
        pos_index = torch.arange(max_turn_length).repeat(len(cid_rep), 1).cuda()    # [B, S]
        cid_reps = torch.stack(cid_reps)
        cid_mask = torch.stack(cid_mask).cuda()
        spk_index = torch.ones(len(cid_rep), max_turn_length).cuda()    # [B, S]
        spk_index[:, ::2] = 0
        spk_index = spk_index.to(torch.long)
        return cid_reps, cid_mask, pos_index, spk_index  # [B, S, E], [B, S], [B, S]
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid.shape[0]
        
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # [S, E], [10, E]
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)
        
        pos_embd = self.position_embd(pos_index)
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [1, S, E]

        cid_rep += cid_rep_base
        cid_rep = cid_rep[:, cid_turn_length-1, :]    # [1, E]
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()    # [10] 
        return dot_product

    def forward(self, cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, recover_mapping)
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)

        # Transformer Encoder
        pos_embd = self.position_embd(pos_index)    # [B, S, E]
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [B, S, E]

        cid_rep += cid_rep_base

        last_utterance = []
        for i in range(len(cid_turn_length)):
            c = cid_rep[i]
            p = cid_turn_length[i]
            last_utterance.append(c[p-1, :])
        cid_rep = torch.stack(last_utterance)    # [B_c, E]

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        loss_1 = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_1.sum(dim=1)).mean()
        return loss, acc
