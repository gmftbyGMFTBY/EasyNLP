from model.utils import *
from .dual_bert_hier import *

class HashBERTDualHierEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualHierEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.hash_loss_matrix_scale = args['hash_loss_matrix_scale']
        self.kl_loss_scale = args['kl_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.temp = args['temp']
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.base_model = BERTDualHierarchicalTrsMVColBERTEncoder(**args) 

        # deep hashing module
        inpt_size = 768 
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
        self.kl_loss = torch.nn.MSELoss()

    def compact_binary_vectors(self, ids):
        # ids: [B, D]
        ids = ids.cpu().numpy().astype('int')
        ids = np.split(ids, int(ids.shape[-1]/8), axis=-1)
        ids = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in ids]    
            ).transpose().astype('uint8')
        )
        return ids

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, ids_mask)
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
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
        self.base_model.eval()
        cid_rep = self.ctx_encoder(ids, ids_mask)
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        return hash_code

    @torch.no_grad()
    def predict(self, batch):
        self.base_model.eval()
        self.ctx_hash_encoder.eval()
        self.can_hash_encoder.eval()

        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        turn_length = batch['turn_length']

        cid = cid.squeeze(0)
        cid_mask = cid_mask.squeeze(0)
        batch_size = rid.shape[0]
            
        cid_reps, rid_rep, rid_mask, pos_index = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.base_model.get_context_level_rep(cid_reps, pos_index)
        cid_turn_length = cid_mask_.sum(dim=-1)

        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        distance = self.get_dot_product(ctx_hash_code, can_hash_code, cid_mask_, rid_mask).squeeze(0)
        return distance

    def get_dot_product(self, cid_rep, rid_rep, cid_mask, rid_mask):
        # cid_rep: [B_c, S_c, E]; rid_rep: [B_r, S_r, E]
        bsz_c, seqlen_c, _ = cid_rep.size()
        bsz_r, seqlen_r, _ = rid_rep.size()
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())
        
        # convert the dot product to the negative hamming distance
        dp = -0.5 * (self.hash_code_size - dp)

        # masking
        cid_mask = cid_mask.to(torch.long)
        cid_mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask))
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf

        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        dp = dp.max(dim=-1)[0]
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp

    def get_dot_product_train(
        self, cid_rep, rid_rep, cid_mask, rid_mask,
        cid_rep_f, rid_rep_f
    ):
        # cid_rep: [B_c, S_c, E]; rid_rep: [B_r, S_r, E]
        bsz_c, seqlen_c, _ = cid_rep.size()
        bsz_r, seqlen_r, _ = rid_rep.size()
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        cid_rep_f = cid_rep_f.reshape(bsz_c*seqlen_c, -1)
        rid_rep_f = rid_rep_f.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())
        dp_f = torch.matmul(cid_rep_f, rid_rep_f.t())
        
        # masking
        cid_mask = cid_mask.to(torch.long)
        cid_mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask))
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf
        dp_f[mask == 0] = -np.inf

        # [B_c*S_c, B_r, S_r]
        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        dp_f = torch.stack(torch.split(dp_f, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        dp_f_index = dp.max(dim=-1)[1].unsqueeze(2)    # [B_c*S_c, B_r, 1]
        dp = torch.gather(dp, 2, dp_f_index).squeeze(2)    # [B_c*S_c, B_r]
        # dp = dp.max(dim=-1)[0]    # [B_c*S_c, B_r]
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        # dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp

    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        turn_length = batch['turn_length']

        with torch.no_grad():
            cid_reps, rid_rep, rid_mask, pos_index = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
            cid_rep, cid_mask_ = self.base_model.get_context_level_rep(cid_reps, pos_index)
            cid_mask_length = cid_mask_.to(torch.long)
            cid_mask_length = torch.where(cid_mask_length == 0, torch.ones_like(cid_mask_length), torch.zeros_like(cid_mask_length))
            cid_turn_length = cid_mask_length.sum(dim=-1)
        
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B_c, S_c, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B_r, S_r, H]
        ctx_hash_code_re = self.ctx_hash_decoder(ctx_hash_code)    # [B_c, S_c, E]
        can_hash_code_re = self.can_hash_decoder(can_hash_code)    # [B_r, S_r, E]
        
        matrix = self.get_dot_product_train(ctx_hash_code, can_hash_code, cid_mask_, rid_mask, cid_rep, rid_rep)    # [B_c, B_r, S_c]
        matrix_ = self.base_model.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask)

        # ==== MSE Loss ===== #
        kl_loss =  self.kl_loss_scale * (
            self.kl_loss(ctx_hash_code_re.reshape(-1, 768), cid_rep.reshape(-1, 768)) +\
            self.kl_loss(can_hash_code_re.reshape(-1, 768), rid_rep.reshape(-1, 768))
        )
        # ==== Quantization loss ===== #
        ctx_hash_code_h = torch.sign(
            ctx_hash_code.reshape(-1, self.hash_code_size)
        ).detach()
        can_hash_code_h = torch.sign(
            can_hash_code.reshape(-1, self.hash_code_size)
        ).detach()
        quantization_loss = torch.norm(ctx_hash_code.reshape(-1, self.hash_code_size) - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code.reshape(-1, self.hash_code_size) - can_hash_code_h, p=2, dim=1).mean()

        # ===== calculate hash loss ===== #
        batch_size = cid_rep.shape[0]
        r_batch_size = rid_rep.shape[0]
        mask = torch.zeros_like(matrix)
        mask[range(batch_size), range(r_batch_size)] = 1.
        label_matrix = 10 * self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale

        # ===== calculate hamming distance for accuracy ===== #
        hamming_distance = 0.5 * (10 * self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== ref acc ===== #
        ref_acc_num = (matrix_.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size
        return kl_loss, hash_loss, quantization_loss, acc, ref_acc


class LSHHierEncoder(nn.Module):
    
    def __init__(self, **args):
        super(LSHHierEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        model = args['pretrained_model']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.temp = args['temp']
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.base_model = BERTDualHierarchicalTrsMVColBERTEncoder(**args) 
        self.lsh_model = nn.Parameter(torch.randn(768, self.hash_code_size))

    def compact_binary_vectors(self, ids):
        # ids: [B, D]
        ids = ids.cpu().numpy().astype('int')
        ids = np.split(ids, int(ids.shape[-1]/8), axis=-1)
        ids = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in ids]    
            ).transpose().astype('uint8')
        )
        return ids

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

    def get_dot_product(self, cid_rep, rid_rep, cid_mask, rid_mask):
        # cid_rep: [B_c, S_c, E]; rid_rep: [B_r, S_r, E]
        bsz_c, seqlen_c, _ = cid_rep.size()
        bsz_r, seqlen_r, _ = rid_rep.size()
        cid_rep = cid_rep.reshape(bsz_c*seqlen_c, -1)
        rid_rep = rid_rep.reshape(bsz_r*seqlen_r, -1)
        dp = torch.matmul(cid_rep, rid_rep.t())
        
        # convert the dot product to the negative of the hamming distance
        dp = -0.5 * (self.hash_code_size - dp)

        # masking
        cid_mask = cid_mask.to(torch.long)
        cid_mask = torch.where(cid_mask == 0, torch.ones_like(cid_mask), torch.zeros_like(cid_mask))
        cid_mask = cid_mask.reshape(-1, 1).expand(-1, bsz_r*seqlen_r)
        rid_mask = rid_mask.reshape(1, -1).expand(bsz_c*seqlen_c, -1)
        mask = cid_mask * rid_mask
        dp[mask == 0] = -np.inf

        #
        dp = torch.stack(torch.split(dp, seqlen_r, dim=-1), dim=-1).permute(0, 2, 1)
        dp = dp.max(dim=-1)[0]
        dp = torch.where(dp == -np.inf, torch.zeros_like(dp), dp).t()
        dp = torch.stack(torch.split(dp, seqlen_c, dim=-1), dim=-1).permute(0, 2, 1).sum(dim=-1).t()    # [B_c, B_r]
        return dp

    @torch.no_grad()
    def predict(self, batch):
        self.base_model.eval()

        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        turn_length = batch['turn_length']

        batch_size = rid.shape[0]
        cid = cid.squeeze(0)
        cid_mask = cid_mask.squeeze(0)
        
        cid_reps, rid_rep, rid_mask, pos_index = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep, cid_mask_ = self.base_model.get_context_level_rep(cid_reps, pos_index)

        cid_rep, rid_rep = torch.matmul(cid_rep, self.lsh_model), torch.matmul(rid_rep, self.lsh_model)
        cid_rep = torch.sign(cid_rep)    # [B_c, S_c, Hash]
        rid_rep = torch.sign(rid_rep)    # [B_r, S_r, Hash]
        distance = self.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask).squeeze(0)
        return distance
        
    def forward(self, batch):
        pass
