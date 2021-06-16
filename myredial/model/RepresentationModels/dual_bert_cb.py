from model.utils import *

'''dual bert model with codebook'''

class BERTDualCBEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualCBEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        code_size = args['code_size']
        code_num = args['code_num']
        self.book_num = args['book_num']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)

        self.codebooks = nn.ModuleList([
            nn.Linear(code_size, code_num) for _ in range(self.book_num)
        ])
        self.cb_head = nn.Sequential(
            nn.Linear(code_size*self.book_num, code_size),
            nn.ReLU(),
            nn.Linear(code_size, code_size)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(768 + code_size, 768 + code_size),
            nn.ReLU(),
            nn.Linear(768 + code_size, 768 + code_size)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)

        # q-k
        cid_reps = []
        for i in range(self.book_num):
            scores = torch.matmul(cid_rep, self.codebooks[i].weight.transpose(0, 1))
            scores = F.softmax(scores/self.temp, dim=-1)    # [B, N]
            cid_rep_ = torch.matmul(scores, self.codebooks[i].weight)   # [B, E]
            cid_reps.append(cid_rep_)
        code_cid_rep = torch.cat(cid_reps, dim=-1)    # [B, E*10]
        
        rid_reps = []
        for i in range(self.book_num):
            scores = torch.matmul(rid_rep, self.codebooks[i].weight.transpose(0, 1))
            scores = F.softmax(scores/self.temp, dim=-1)    # [B, N]
            rid_rep_ = torch.matmul(scores, self.codebooks[i].weight)    # [B, E]
            rid_reps.append(rid_rep_)
        code_rid_rep = torch.cat(rid_reps, dim=-1)    # [B, E*10]

        code_cid_rep = self.cb_head(code_cid_rep)
        code_rid_rep = self.cb_head(code_rid_rep)

        cid_rep = self.fusion_head(
            torch.cat([cid_rep, code_cid_rep], dim=1)        
        )
        rid_rep = self.fusion_head(
            torch.cat([rid_rep, code_rid_rep], dim=-1)        
        )
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        scores = torch.matmul(rid_rep, self.codebook)    # [B, N]
        scores = F.softmax(scores, dim=-1)    # [B, N]
        rid_rep = torch.matmul(scores, self.codebook.transpose(0, 1))    # [B, E]
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        scores = torch.matmul(cid_rep, self.codebook)    # [B, N]
        scores = F.softmax(scores, dim=-1)    # [B, N]
        cid_rep = torch.matmul(scores, self.codebook.transpose(0, 1))    # [B, E]
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product

        # label smooth loss
        gold = torch.arange(batch_size).cuda()
        loss = self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
