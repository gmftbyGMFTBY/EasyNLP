from model.utils import *

class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    def forward(self, ids, attn_mask):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return embd

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)


class PolyEncoder(nn.Module):
    
    def __init__(self, **args):
        super(PolyEncoder, self).__init__()
        model = args['pretrained_model']
        m = args['poly_m']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.poly_embd = nn.Embedding(m, 768)
        self.m = m
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        poly_query_id = torch.arange(self.m, dtype=torch.long).cuda()
        poly_query = self.poly_embd(poly_query_id)    # [M, E]
        # NOTE: DO NOT CALCULATE THE EMBEDDINGS OF THE [PAD] TOKEN
        # cid_rep: [B, S, E]; [M, E]
        weights = torch.matmul(cid_rep, poly_query.t()).permute(0, 2, 1)    # [B, M, S]
        weights /= np.sqrt(768)
        cid_mask_ = torch.where(cid_mask != 0, torch.zeros_like(cid_mask), torch.ones_like(cid_mask))
        cid_mask_ = cid_mask_ * -1e3
        cid_mask_ = cid_mask_.unsqueeze(1).repeat(1, self.m, 1)    # [B, M, S]
        weights += cid_mask_
        weights = F.softmax(weights, dim=-1)

        cid_rep = torch.bmm(
            weights,     # [B, M, S]
            cid_rep,     # [B, S, E]
        )    # [B, M, E]

        rid_rep = self.can_encoder(rid, rid_mask)
        rid_rep = rid_rep[:, 0, :]    # [B, E]
        # cid_rep: [B, M, E]; rid_rep: [B, E]
        return cid_rep, rid_rep
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_mask = torch.ones(1, len(cid)).cuda()
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, cid_mask, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [M, E]
        # cid_rep/rid_rep: [M, E], [B, E]
        
        # POLY ENCODER ATTENTION
        # [M, E] X [E, S] -> [M, S] -> [S, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1)
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        # [S, M] X [M, E] -> [S, E]
        cid_rep = torch.matmul(weights, cid_rep)
        dot_product = (cid_rep * rid_rep).sum(-1)    # [S]
        return dot_product
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, M, E];
        
        # POLY ENCODER ATTENTION
        # [B, M, E] X [E, B] -> [B, M, B]-> [B, B, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1)    # [B, M, B] -> [B, B, M]
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)    # [B, B, M] X [B, M, E] -> [B, B, E]
        # [B, B, E] x [B, B, E] -> [B, B]
        dot_product = (cid_rep * rid_rep.unsqueeze(0).expand(batch_size, -1, -1)).sum(-1)
        mask = torch.eye(batch_size).cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
