from model.utils import *


class BERTDualTargetHierTrsEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualTargetHierarchicalTrsEncoder, self).__init__()
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
