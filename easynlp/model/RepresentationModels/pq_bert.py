from model.utils import *


class PQEncoder(nn.Module):
    
    def __init__(self, **args):
        super(PQEncoder, self).__init__()
        self.args = args
        model = args['pretrained_model']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, ids_mask)
        rid_rep = torch.matmul(rid_rep, self.lsh_model)
        hash_code = torch.sign(rid_rep)
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        hash_code = torch.from_numpy(hash_code)
        return hash_code
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, ids_mask)
        cid_rep = torch.matmul(cid_rep, self.lsh_model)
        hash_code = torch.sign(cid_rep)
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        return hash_code

    @torch.no_grad()
    def predict(self, batch):
        self.ctx_encoder.eval()
        self.can_encoder.eval()

        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = torch.ones_like(cid), batch['rids_mask']

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        cid_rep, rid_rep = cid_rep.squeeze(dim=0).cpu().numpy(), rid_rep.cpu().numpy()

        X_code = self.pq.encode(vecs=rid_rep)
        
        dt = self.pq.dtable(query=cid_rep)
        dists = dt.adist(codes=X_code)
        dists = -torch.from_numpy(dists).cuda()
        return dists

    @torch.no_grad()
    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        with torch.no_grad():
            cid_rep = self.ctx_encoder(cid, cid_mask)
            rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)
        reps = torch.cat([cid_rep, rid_rep], dim=0)
        return reps
