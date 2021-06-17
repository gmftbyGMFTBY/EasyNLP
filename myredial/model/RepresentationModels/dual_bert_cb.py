from model.utils import *

'''dual bert model with codebook'''

class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

    def forward(self, ids, attn_mask):
        embds = self.model(ids, attention_mask=attn_mask)[0]
        return embds    # [B, S, E]

    def load_bert_model(self, state_dict):
        # load the post train checkpoint from BERT-FP (NAACL 2021)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)


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

        self.queries = nn.Embedding(self.book_num, code_size)
        self.register_buffer('index', torch.arange(self.book_num)) 

    def query_inf(self, reps, mask):
        # reps: [B, S, E]; mask: [B, S]
        queries = self.queries(self.index)    # [N, E] 
        scores = torch.matmul(reps, queries.transpose(0, 1))
        scores = scores.permute(0, 2, 1)     # [B, N, S]

        padding_mask = torch.where(
            mask == 0, 
            torch.ones_like(mask), 
            torch.zeros_like(mask)
        )    # [B, S] 
        padding_mask = padding_mask.unsqueeze(1).repeat(1, self.book_num, 1).to(torch.float)    # [B, N, S]
        padding_mask *= -1e3
        scores += padding_mask    # [B, N, S]
        scores = F.softmax(scores, dim=-1)

        # extract
        embds = torch.bmm(scores, reps)    # [B, N, E]
        return embds

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, S, E]
        rid_rep = self.can_encoder(rid, rid_mask)

        cid_reps = self.query_inf(cid_rep, cid_mask)    # [B, N, E]
        rid_reps = self.query_inf(rid_rep, rid_mask)

        # q-k
        cid_repss = []
        for i in range(self.book_num):
            scores = torch.matmul(
                cid_reps[:, i, :], 
                self.codebooks[i].weight.transpose(0, 1)
            )
            scores = F.softmax(scores/self.temp, dim=-1)    # [B, N]
            cid_rep_ = torch.matmul(scores, self.codebooks[i].weight)   # [B, E]
            cid_repss.append(cid_rep_)
        code_cid_rep = torch.cat(cid_repss, dim=-1)    # [B, E*10]
        
        rid_repss = []
        for i in range(self.book_num):
            scores = torch.matmul(
                rid_reps[:, i, :], 
                self.codebooks[i].weight.transpose(0, 1)
            )
            scores = F.softmax(scores/self.temp, dim=-1)    # [B, N]
            rid_rep_ = torch.matmul(scores, self.codebooks[i].weight)    # [B, E]
            rid_repss.append(rid_rep_)
        code_rid_rep = torch.cat(rid_repss, dim=-1)    # [B, E*10]

        code_cid_rep = self.cb_head(code_cid_rep)
        code_rid_rep = self.cb_head(code_rid_rep)

        cid_rep = self.fusion_head(
            torch.cat([cid_reps.mean(dim=1), code_cid_rep], dim=-1) 
        )
        rid_rep = self.fusion_head(
            torch.cat([rid_reps.mean(dim=1), code_rid_rep], dim=-1) 
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
        cid = cid.unsqueeze(0)
        cid_rep, rid_rep = self._encode(cid, rid, torch.ones_like(cid), rid_mask)
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
