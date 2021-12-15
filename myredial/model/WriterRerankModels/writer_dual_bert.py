from model.utils import *
from dataloader.util_func import *

class WriterDualEncoder(nn.Module):

    def __init__(self, **args):
        super(WriterDualEncoder, self).__init__()
        self.ctx_encoder = AutoBertEmbedding(model=args['pretrained_model'])
        self.can_encoder = AutoBertEmbedding(model=args['pretrained_model'])

        # compatible with the former model
        # self.ctx_encoder.model.resize_token_embeddings(self.ctx_encoder.model.config.vocab_size + 1)
        # self.can_encoder.model.resize_token_embeddings(self.can_encoder.model.config.vocab_size + 1)

        self.vocab = AutoTokenizer.from_pretrained(args['pretrained_model'])
        self.cls, self.pad, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[PAD]', '[SEP]'])
        self.topk = args['inference_time'] + 1
        self.args = args

    def batchify(self, batch):
        batch['rids'] = [batch['rids'][0] + batch['erids']]
        context, responses = batch['cids'], batch['rids']
        cids, rids = [], []
        for c, rs in zip(context, responses):
            c = torch.LongTensor([self.cls] + c[-(self.args['ctx_max_len']-2):] + [self.sep])
            cids.append(c)
            for r in rs:
                r = torch.LongTensor([self.cls] + r[:self.args['res_max_len']-2] + [self.sep])
                rids.append(r)
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        cids_mask = generate_mask(cids)
        rids_mask = generate_mask(rids)
        cids, rids, cids_mask, rids_mask = to_cuda(cids, rids, cids_mask, rids_mask)
        return cids, rids, cids_mask, rids_mask

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cids, rids, cids_mask, rids_mask = self.batchify(batch)
        cid_rep, rid_rep = self._encode(cids, rids, cids_mask, rids_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        # api normalization
        dot_product /= np.sqrt(768)
        dot_product = (dot_product - dot_product.min()) / (1e-3 + dot_product.max() - dot_product.min())
        return dot_product
    
    def forward(self, batch):
        cids, rids, cids_mask, rids_mask = self.batchify(batch)
        batch_size = len(cids)
        r_batch_size = len(rids)
        cid_rep, rid_rep = self._encode(cids, rids, cids_mask, rids_mask)
        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, r_batch_size, self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (dot_product.max(dim=-1)[1] == torch.arange(0, r_batch_size, self.topk).cuda()).to(torch.float).mean().item()
        return loss, acc
