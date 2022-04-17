from model.utils import *


class BERTDualTargetHierarchicalTrsMVEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualTargetHierarchicalTrsMVEncoder, self).__init__()
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

    @torch.no_grad()
    def get_ctx_embedding(self, ids, attn_mask):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        self.fusion_layer.eval()
        self.squeeze_layer.eval()
        cid_rep = self.ctx_encoder(ids, attn_mask, hidden=True)    # [B, S, E]
        cid_rep = F.normalize(cid_rep, dim=-1)
        cid_length = attn_mask.sum(dim=-1).tolist()    # [B]
        lengths = [min(l, self.mv_num) for l in cid_length]
        cid_rep = [cid_rep_[:l, :] for cid_rep_, l in zip(cid_rep, lengths)]    # B*[T, E]
        return cid_rep

    @torch.no_grad()
    def get_ctx_level_embedding(self, cid_rep, cache, cache_sequence):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        self.fusion_layer.eval()
        self.squeeze_layer.eval()

        empty_label = True if len(cache) == 0 else False
        if empty_label is False:
            cache_ = torch.cat(cache)    # [T, E]
        reps = []
        reps_mask = []
        selected_index = []
        sequences = []
        max_subseq_length = max([len(i) for i in cid_rep])
        ipdb.set_trace()
        last_cid_rep = []
        for c in cid_rep:
            last_cid_rep.append(c[-1])
            if empty_label is False:
                selected_index = len(cache_) + len(c) - 1
            else:
                selected_index = len(c) - 1
            if len(c) < max_subseq_length:
                c = torch.cat([c, torch.zeros(max_subseq_length - len(c), 768).cuda()], dim=0)

            if empty_label is False:
                reps_ = torch.cat([cache_, c], dim=0)   # [S, E]
                sequences = cache_sequence + [cache_sequence[-1] + 1] * (len(c))
            else:
                reps_ = c 
                sequences = [0] * (len(c))
            m = torch.tensor([False] * len(reps_) + [True] * (max_subseq_length - len(reps_))).to(torch.bool)
            reps_mask.append(m)
            reps.append(reps_)
        last_cid_rep = torch.stack(last_cid_rep)
        reps_mask = torch.stack(reps_mask).cuda()
        reps = torch.stack(reps)
        seqlen_index = torch.LongTensor(sequences).cuda()   # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
        reps += pos_embd

        reps_ = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=reps_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        reps_ = reps_[:, 0, :]
        cid_rep = self.squeeze_layer(
            torch.cat([reps_, reps[:, 0, :]], dim=-1)        
        )    # [B, E]
        cid_rep = F.normalize(cid_rep, dim=-1)
        return cid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        self.fusion_layer.eval()
        self.squeeze_layer.eval()
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = F.normalize(rid_rep, dim=-1)
        return rid_rep

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask, hidden=True)    # [B, S, E]
        cid_length = cids_mask.sum(dim=-1).tolist()    # [B]
        lengths = [min(l, self.mv_num) for l in cid_length]
        cid_rep = [cid_rep_[:l, :] for cid_rep_, l in zip(cid_rep, lengths)]    # B*[T, E]

        cid_reps = []
        sequence = []
        index = 0
        for t in turn_length:
            chunk = cid_rep[index:index+t]
            chunk_l = [len(i) for i in chunk]
            chunk = torch.cat(chunk, dim=0)   # [S, E]
            cid_reps.append(chunk)
            sequence.append(
                list(chain(*[[idx]*i for idx, i in enumerate(chunk_l)]))
            )
            index += t
        reps, reps_mask, sequences = [], [], []    # [B, S]
        last_cid_rep = []
        max_seq_length = max([len(i) for i in sequence])
        select_index = []
        for cid_rep, s_ in zip(cid_reps, sequence):
            select_index.append(len(s_) - 1)
            last_cid_rep.append(cid_rep[-1])
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_seq_length - len(cid_rep))).to(torch.bool)
            reps_mask.append(m)
            if len(cid_rep) < max_seq_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_seq_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
            sequences.append(s_ + [0] * (max_seq_length - len(s_)))
        reps = torch.stack(reps)    # [B, S, E]
        reps_mask = torch.stack(reps_mask).cuda()    # [B, S]
        seqlen_index = torch.LongTensor(sequences).cuda()   # [B, S]
        pos_embd = self.position_embedding(seqlen_index)    # [B, S, E]
        reps += pos_embd

        reps_ = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=reps_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        reps_ = reps_[:, 0, :]
        cid_rep = self.squeeze_layer(
            torch.cat([reps_, reps[:, 0, :]], dim=-1)        
        )    # [B, E]

        # response
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, E]
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        self.fusion_layer.eval()
        self.squeeze_layer.eval()
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product /= self.args['temp']
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
