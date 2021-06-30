from model.utils import *

class BERTDualCompEncoder(nn.Module):

    '''This model needs the gray(hard negative) samples, which cannot be used for recall'''
    
    def __init__(self, **args):
        super(BERTDualCompEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        self.gray_num = args['gray_cand_num']
        nhead = args['nhead']
        dim_feedforward = args['dim_feedforward']
        dropout = args['dropout']
        num_encoder_layers = args['num_encoder_layers']

        # ====== Model ====== #
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        hidden_size = self.ctx_encoder.model.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size*2, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(2*hidden_size)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        self.trs_head = nn.Sequential(
            self.trs_encoder,
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2, hidden_size),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 3),
        )


    def _encode(self, cid, rid, cid_mask, rid_mask):
        # G = B_r*gray_num
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [G, E]
        b_c, g = len(cid_rep), len(rid)

        rid_reps = rid_rep.unsqueeze(0).repeat(b_c, 1, 1)    # [B, G, E]
        cid_reps = cid_rep.unsqueeze(1).repeat(1, g, 1)    # [B, G, E]
        for_comp = torch.cat([rid_reps, cid_reps], dim=-1)   # [B, G, 2*E]
        comp_reps = self.trs_head(for_comp.permute(1, 0, 2)).permute(1, 0, 2)    # [B, G, E] 

        # fusion
        reps = self.cls_head(
            torch.cat([cid_reps, rid_reps, comp_reps], dim=-1)
        )    # [B, G, 3*E] -> [B, G, E]

        # combine and binary classification
        # [B, G, E] -> [B, G, G]
        matrix, label = [], []
        for idx, sample in enumerate(reps):
            # sample: [G, E] -> [G, G, E]
            sample_1 = sample.unsqueeze(1).repeat(1, g, 1)    # [G, G', E]
            sample_2 = sample.unsqueeze(0).repeat(g, 1, 1)    # [G', G, E]
            sample = torch.cat([sample_1, sample_2], dim=-1)     # [G, G, 2*E]
            matrix_ = self.decision_head(sample)    # [G, G, 3]
            matrix.append(matrix_)

            # label:
            # 2: front better; 1: hard to tell; 0: front worse
            label_ = torch.zeros(g, g).cuda()     # [G, G]
            begin = idx*(1+self.gray_num)
            end = (idx+1)*(1+self.gray_num)
            for i in range(g):
                if i == begin:
                    # 1. positive:
                    label_[begin, :] = 2.
                elif i in range(begin+1, end):
                    # 2. hard negative:
                    before = list(range(0, begin))
                    after = list(range(end, g))
                    inner = list(range(begin+1, end))
                    label_[i, before] = 2.
                    label_[i, after] = 2.
                    label_[i, inner] = 1.
                    label_[i, begin] = 0.
                else:
                    # random negative
                    before = list(range(0, begin))
                    after = list(range(end, g))
                    label_[i, before] = 1.
                    label_[i, after] = 1.
            # diag
            label_[range(g), range(g)] = 1.
            label.append(label_)
        matrix = torch.stack(matrix)    # [B, G, G, 3]
        label = torch.stack(label).to(torch.long)    # [B, G, G]
        return matrix, label

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid)

        batch_size = rid.shape[0]
        matrix, _ = self._encode(cid, rid, cid_mask, rid_mask).squeeze(0)     # [B, G, G, 3] -> [G, G, 3]
        label = F.softmax(matrix, dim=-1).max(dim=-1)[1]    # [G, G]
        # reconstruct the order
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        b_c, b_r = len(cid), int(len(rid)//(self.gray_num+1))
        assert b_c == b_r
        matrix, label = self._encode(cid, rid, cid_mask, rid_mask)    # [B, G, G, 3], [B, G, G]
        matrix = matrix.view(-1, 3)
        label = label.view(-1)

        num_0 = (label == 0).sum()
        num_1 = (label == 1).sum()
        num_2 = (label == 2).sum()
        w_0 = (len(label) - num_0) / len(label)
        w_1 = (len(label) - num_1) / len(label)
        w_2 = (len(label) - num_2) / len(label)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_0, w_1, w_2]).cuda())
        loss = criterion(matrix, label)

        # acc
        acc = (F.softmax(matrix, dim=-1).max(dim=-1)[1] == label).to(torch.float).mean().item()
        return loss, acc
