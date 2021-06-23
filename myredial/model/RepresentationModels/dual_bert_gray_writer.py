from model.utils import *

    
class BERTDualGrayFullEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualGrayFullEncoder, self).__init__()
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num']

        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.args = args

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
        context = batch['context']
        responses = batch['responses']
        cid, cid_mask = self.totensor([context], ctx=True)
        rid, rid_mask = self.totensor(responses, ctx=False)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids

    def totensor(self, texts, ctx=True):
        items = self.vocab.batch_encode_plus(texts)['input_ids']
        if ctx:
            ids = [torch.LongTensor(self._length_limit(i)) for i in items]
        else:
            ids = [torch.LongTensor(self._length_limit_res(i)) for i in items]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, mask = ids.cuda(), mask.cuda()
        return ids, mask
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
    
    def forward(self, batch):
        context = batch['context']
        responses = batch['responses']
        cid, cid_mask = self.totensor(context, ctx=True)
        rid, rid_mask = self.totensor(responses, ctx=False)

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, 10*B]
        dot_product /= np.sqrt(768)

        mask = torch.zeros_like(dot_product).cuda()
        mask[torch.arange(batch_size), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        # loss
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        return loss, acc
