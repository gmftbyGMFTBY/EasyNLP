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
        
        self.beta_gamma = self.args['beta_gamma']
        self.criterion = nn.MarginRankingLoss(margin=2.)
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

    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        turn_length = batch['turn_length']
        batch_size = len(cid)

        with torch.no_grad():
            cid_reps, rid_rep, rid_mask, pos_index = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
            cid_rep, cid_mask_ = self.base_model.get_context_level_rep(cid_reps, pos_index)
            cid_mask_length = cid_mask_.to(torch.long)
            cid_mask_length = torch.where(cid_mask_length == 0, torch.ones_like(cid_mask_length), torch.zeros_like(cid_mask_length))
            cid_turn_length = cid_mask_length.sum(dim=-1)
        
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B_c, S_c, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B_r, S_r, H]
        beta = np.sqrt(batch['current_step'] * self.beta_gamma + 1)
        ctx_hash_code = torch.tanh(ctx_hash_code * beta)
        can_hash_code = torch.tanh(can_hash_code * beta)
        
        matrix = self.base_model.get_dot_product(ctx_hash_code, can_hash_code, cid_mask_, rid_mask)
        matrix_ = self.base_model.get_dot_product(cid_rep, rid_rep, cid_mask_, rid_mask)

        # ===== calculate hash loss ===== #
        pos_reps, neg_reps = [], []
        for idx, line in enumerate(matrix):
            pos_reps.append(matrix[idx, idx])
            random_index = list(range(len(matrix)))
            random_index.remove(idx)
            random_index = random.choice(random_index)
            neg_reps.append(matrix[idx, random_index])
        pos_reps = torch.stack(pos_reps)
        neg_reps = torch.stack(neg_reps)
        hash_loss = self.criterion(pos_reps, neg_reps, torch.ones_like(pos_reps))

        # ===== calculate hamming distance for accuracy ===== #
        hamming_distance = 0.5 * (10 * self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== ref acc ===== #
        ref_acc_num = (matrix_.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size
        return hash_loss, acc, ref_acc, beta


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


class HashBERTDualHierNoColBERTEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualHierNoColBERTEncoder, self).__init__()
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
        self.base_model = BERTDualHierarchicalTrsMVEncoder(**args) 

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
        self.base_model.eval()
        self.can_hash_encoder.eval()
        rid_rep = self.base_model.can_encoder(ids, ids_mask)
        rid_rep = F.normalize(rid_rep, dim=-1)
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
    def get_ctx(self, ids, ids_mask, turn_length):
        self.base_model.eval()
        self.ctx_hash_encoder.eval()
        cid_rep = self.base_model.ctx_encoder(ids, ids_mask, hidden=True)
        cid_rep = cid_rep[:, :self.base_model.mv_num, :].reshape(-1, 768)
        new_turn_length = [i*self.base_model.mv_num for i in turn_length]
        cid_reps = torch.split(cid_rep, new_turn_length)
        cid_rep = self.base_model.get_context_level_rep(cid_reps, turn_length)
        cid_rep = F.normalize(cid_rep, dim=-1)
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        hash_code = torch.from_numpy(hash_code).cuda()
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

        cid_rep, rid_rep = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.base_model.get_context_level_rep(cid_rep, turn_length)
        cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)

        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t()).squeeze(dim=0)
        distance = -0.5 * (self.hash_code_size - matrix)
        return distance

    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        turn_length = batch['turn_length']
        batch_size = len(rid)

        with torch.no_grad():
            cid_rep, rid_rep = self.base_model._encode(cid, rid, cid_mask, rid_mask, turn_length)
            cid_rep = self.base_model.get_context_level_rep(cid_rep, turn_length)
            cid_rep, rid_rep = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B_c, S_c, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B_r, S_r, H]
        ctx_hash_code_re = self.ctx_hash_decoder(ctx_hash_code)    # [B_c, S_c, H]
        can_hash_code_re = self.can_hash_decoder(can_hash_code)    # [B_r, S_r, H]
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()

        # kl_loss = self.kl_loss(ctx_hash_code_re, cid_rep) + self.kl_loss(can_hash_code_re, rid_rep)
        # kl_loss *= self.kl_loss_scale

        # quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()

        matrix = torch.matmul(ctx_hash_code, can_hash_code.t())
        matrix_ = torch.matmul(cid_rep, rid_rep.T)

        # ===== calculate hash loss ===== #
        mask = torch.zeros_like(matrix)
        mask[range(batch_size), range(batch_size)] = 1.
        label_matrix = self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean()
        # hash_loss += quantization_loss + kl_loss

        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(len(rid))).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== ref acc ===== #
        ref_acc_num = (matrix_.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size
        return hash_loss, acc, ref_acc, 0

