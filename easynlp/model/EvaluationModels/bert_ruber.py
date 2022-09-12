from model.utils import *

class BERTRUBERModel(nn.Module):

    def __init__(self, **args):
        super(BERTRUBERModel, self).__init__()
        model = args['pretrained_model']
        self.encoder = BertEmbedding(model=model, add_tokens=1)
        
        self.M = nn.Parameter(torch.rand(768, 768))
        self.layer1 = nn.Linear(768 * 2 + 1, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
        self.opt = nn.Linear(128, 2)
        self.criterion = nn.CrossEntropyLoss()

        self.args = args

    @torch.no_grad()
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.encoder(cid, cid_mask)
        rid_rep = self.encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        qh, rh = self._encode(cid, rid, cid_mask, rid_mask)
        qh = qh.unsqueeze(1)    # [B, 1, 768]
        rh = rh.unsqueeze(2)    # [B, 768, 1]
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = torch.tanh(self.layer1(linear))
        linear = torch.tanh(self.layer2(linear))
        linear = torch.tanh(self.layer3(linear))
        dot_product = self.opt(linear)  # [B, 2]
        return dot_product[:, 1]

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        label = batch['label']

        qh, rh = self._encode(cid, rid, cid_mask, rid_mask)
        
        qh = qh.unsqueeze(1)    # [B, 1, 768]
        rh = rh.unsqueeze(2)    # [B, 768, 1]
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = torch.tanh(self.layer1(linear))
        linear = torch.tanh(self.layer2(linear))
        linear = torch.tanh(self.layer3(linear))
        dot_product = self.opt(linear)  # [B, 2]

        loss = self.criterion(dot_product, label)

        acc = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == label).float().mean().item()
        return loss, acc
