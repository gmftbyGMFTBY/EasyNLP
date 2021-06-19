from model.utils import *

class HashBERTDualEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num']
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        gpu_ids = str(args['trainable_bert_layers'])
        self.trainable_layers = [f'encoder.layer.{i}' for i in gpu_ids.split(',')]
        inpt_size = self.ctx_encoder.model.config.hidden_size
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.ctx_hash_decoder = nn.Sequential(
            nn.Linear(self.hash_code_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, inpt_size),
        )
        self.can_hash_decoder = nn.Sequential(
            nn.Linear(self.hash_code_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, inpt_size),
        )

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        rid_rep = self.can_encoder(ids, ids_mask)
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        return hash_code
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        cid_rep = self.ctx_encoder(ids, ids_mask)
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        return hash_code

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
    
    @torch.no_grad()
    def predict(self, batch):
        context = batch['context']
        responses = batch['responses']
        cid, cid_mask = self.totensor([context], ctx=True)
        rid, rid_mask = self.totensor(responses, ctx=False)

        cid_rep = self.ctx_encode(cid.unsqueeze(0), None)
        rid_rep = self.can_encoder(rid, rid_mask)

        cid_rep = cid.squeeze(0)
        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(cid_hash_code, can_hash_code.t())    # [B]
        distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        context = batch['context']
        responses = batch['responses']
        cid, cid_mask = self.totensor(context, ctx=True)
        rid, rid_mask = self.totensor(responses, ctx=False)

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        
        batch_size = cid_rep.shape[0]

        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B*gray, H]
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)
        rid_rep_recons = self.can_hash_decoder(can_hash_code)

        # ===== calculate preserved loss ===== #
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).mean() 
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)    # [B, B*H] similarity matrix
        mask = torch.zeros_like(matrix)
        mask[torch.arange(batch_size), torch.arange(0, len(rid_rep), self.gray_num+1)] = 1.
        label_matrix = self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== calculate the cl loss ======
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        dot_product /= np.sqrt(768)
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # ===== calculate the ref acc =====
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        ref_acc = acc_num / batch_size
        return loss, preserved_loss, hash_loss, quantization_loss, ref_acc, acc
