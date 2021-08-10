from model.utils import *

class BERTDualCompareEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualCompareEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.comp_train_size = args['comp_train_size']
        self.args = args
        self.alpha = args['alpha']
        p = args['dropout']

        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.head = nn.Sequential(
            nn.Linear(768*3, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        bsz_c, bsz_r = len(cid), len(rid)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)

        # compare 
        tickets = []
        reps = []
        for i_r1 in range(bsz_r):
            for i_r2 in range(bsz_r):
                if i_r1 < i_r2:
                    tickets.append((i_r1, i_r2))
                    reps.append(
                        torch.cat([cid_rep[0], rid_rep[i_r1], rid_rep[i_r2]])    # [768*3]        
                    )
        reps = self.head(torch.stack(reps)).squeeze(1)    # [K]
        reps = torch.sigmoid(reps)    # [K]
        chain = {i: [] for i in range(bsz_r)} 
        for (i, j), l in zip(tickets, reps):
            if l.item() >= 0.5 + self.args['threshold']:
                chain[i].append(j)
            elif l.item() < 0.5 - self.args['threshold']:
                chain[j].append(i)
            else:
                # cannot decide, use the the dot product scores
                if dot_product[i] >= dot_product[j]:
                    chain[i].append(j)
                else:
                    chain[j].append(i)
        ipdb.set_trace()
        scores = self.generate_scores(chain)
        return torch.tensor(scores)

    def generate_one_batch_compare(self, cid_rep, rid_rep):
        rep, label = [], []
        bsz = len(cid_rep)
        random_idx = torch.randperm(bsz)
        rid_rep_ = rid_rep[random_idx]
        for i in range(bsz):
            if random.random() > 0.5:
                # good left and bad right, label: 2
                rep.append(torch.cat([cid_rep[i], rid_rep[i], rid_rep_[i]]))
                label.append(1)
            else:
                # good right and bad left, label: 1
                rep.append(torch.cat([cid_rep[i], rid_rep_[i], rid_rep[i]]))
                label.append(0)
        return rep, label

    def comp_cls(self, cid_rep, rid_rep):
        bsz, label = len(cid_rep), []
        # good vs. bad
        rep, label = [], []
        for i in range(self.comp_train_size):
            rep_, label_ = self.generate_one_batch_compare(cid_rep, rid_rep)
            rep.extend(rep_)
            label.extend(label_)
        rep = self.head(torch.stack(rep)).squeeze(1)    # [B*K]
        # random shuffle
        random_idx = torch.randperm(len(rep))
        rep = torch.stack([rep[i] for i in random_idx])
        label = [label[i] for i in random_idx]
        label = torch.tensor(label).cuda()
        loss = self.criterion(rep, label.float())
        # acc
        acc = ((rep > 0.5).to(torch.float) == label).sum().item()
        acc /= len(label)
        return loss*self.alpha, acc
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B] or [B, 2*B]
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # compare loss
        loss2, acc = self.comp_cls(cid_rep, rid_rep)
        loss += loss2
        return loss, acc
    
    def generate_scores(self, edges):
        # len(edges) = the number of the vertices
        num = len(edges)
        g = Graph(num)
        for i, item_list in edges.items():
            for j in item_list:
                g.addEdge(i, j)
        rest = g.topologicalSort()
        scores = list(reversed(range(num)))
        scores = [(i, j) for i, j in zip(rest, scores)]
        scores = sorted(scores, key=lambda x:x[0])
        scores = [j for i, j in scores]
        return scores
