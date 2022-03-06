from model.utils import *
from model.LanguageModels import *

class HashBERTDualEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num'] + 1
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.hash_loss_matrix_scale = args['hash_loss_matrix_scale']
        self.kl_loss_scale = args['kl_loss_scale']
        self.dis_loss_scale = args['dis_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.temp = args['temp']
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
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

        self.kl_loss = torch.nn.MSELoss()
        self.dis_loss = torch.nn.KLDivLoss()

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
        self.ctx_encoder.eval()
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
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        self.ctx_hash_encoder.eval()
        self.can_hash_encoder.eval()

        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = torch.ones_like(cid), batch['rids_mask']

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)

        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t()).squeeze(0)    # [B]
        # minimal distance -> better performance 
        distance = -0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        
        batch_size = cid_rep.shape[0]
        r_batch_size = rid_rep.shape[0]

        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B*gray, H]
        ctx_hash_code_re = self.ctx_hash_decoder(ctx_hash_code)    # [B, H]
        can_hash_code_re = self.can_hash_decoder(can_hash_code)    # [B*gray, H]

        # ===== MSE Loss ===== #
        # kl_loss = self.kl_loss(ctx_hash_code_re, cid_rep) + self.kl_loss(can_hash_code_re, rid_rep)
        # kl_loss *= self.kl_loss_scale
        # kl_loss = torch.tensor(0.)
        cid_rep_, rid_rep_ = F.normalize(cid_rep, dim=-1), F.normalize(rid_rep, dim=-1)
        dot_product = torch.matmul(cid_rep_, rid_rep_.t())
        dot_product /= self.temp
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        kl_loss = (-loss_.sum(dim=1)).mean()

        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)   # [B, B*H] similarity matrix
        mask = torch.zeros_like(matrix)
        mask[range(batch_size), range(r_batch_size)] = 1.
        label_matrix = self.hash_loss_matrix_scale * self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== ref acc ===== #
        dp = torch.matmul(cid_rep, rid_rep.t())
        ref_acc_num = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size

        return kl_loss, hash_loss, quantization_loss, acc, ref_acc



class HashBERTSupDualEncoder(nn.Module):

    '''hash bert with supervised information:
        1. tf-idf'''
    
    def __init__(self, **args):
        super(HashBERTSupDualEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num'] + 1
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.hash_loss_matrix_scale = args['hash_loss_matrix_scale']
        self.kl_loss_scale = args['kl_loss_scale']
        self.dis_loss_scale = args['dis_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.tfidf_hidden_size = args['tfidf_hidden_size']
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        # TF-IDF Model
        self.tfidf = TFIDFModel(**args)
        self.tfidf2dense = nn.Sequential(
            nn.Linear(self.tfidf.vocab_size, self.tfidf_hidden_size*2),
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.tfidf_hidden_size*2, self.tfidf_hidden_size)
        )

        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        inpt_size = self.ctx_encoder.model.config.hidden_size
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size+self.tfidf_hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size+self.tfidf_hidden_size, self.hidden_size),
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
        self.dis_loss = torch.nn.KLDivLoss()

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
        self.ctx_encoder.eval()
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
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = torch.ones_like(cid), batch['rids_mask']
        ctext, rtext = batch['ctext'], batch['rtext']
        
        # tf-idf embeddings
        tfidf_cid_rep = self.tfidf.predict(ctext)
        tfidf_rid_rep = self.tfidf.predict(rtext)
        tfidf_cid_rep = self.tfidf2dense(tfidf_cid_rep)
        tfidf_rid_rep = self.tfidf2dense(tfidf_rid_rep)

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep_ = torch.cat([cid_rep, tfidf_cid_rep], dim=-1)
        rid_rep_ = torch.cat([rid_rep, tfidf_rid_rep], dim=-1)

        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep_))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep_))    # [B, Hash]
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t()).squeeze(0)    # [B]
        # minimal distance -> better performance 
        distance = -0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']
        ctext, rtext = batch['ctext'], batch['rtext']
        ctext = [c.replace('[SEP]', '') for c in ctext]

        # tf-idf embeddings
        tfidf_cid_rep = self.tfidf.predict(ctext)
        tfidf_rid_rep = self.tfidf.predict(rtext)
        tfidf_cid_rep = self.tfidf2dense(tfidf_cid_rep)
        tfidf_rid_rep = self.tfidf2dense(tfidf_rid_rep)

        with torch.no_grad():
            cid_rep = self.ctx_encoder(cid, cid_mask)
            rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep_ = torch.cat([cid_rep, tfidf_cid_rep], dim=-1)
        rid_rep_ = torch.cat([rid_rep, tfidf_rid_rep], dim=-1)
        
        batch_size = cid_rep.shape[0]
        r_batch_size = rid_rep.shape[0]

        ctx_hash_code = self.ctx_hash_encoder(cid_rep_)    # [B, H]
        can_hash_code = self.can_hash_encoder(rid_rep_)    # [B*gray, H]
        ctx_hash_code_re = self.ctx_hash_decoder(ctx_hash_code)    # [B, H]
        can_hash_code_re = self.can_hash_decoder(can_hash_code)    # [B*gray, H]

        # ===== MSE Loss ===== #
        kl_loss = self.kl_loss(ctx_hash_code_re, cid_rep) + self.kl_loss(can_hash_code_re, rid_rep)
        kl_loss *= self.kl_loss_scale

        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)   # [B, B*H] similarity matrix
        mask = torch.zeros_like(matrix)
        mask[range(batch_size), range(r_batch_size)] = 1.
        label_matrix = self.hash_loss_matrix_scale * self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        acc = acc_num / batch_size

        # ===== ref acc ===== #
        dp = torch.matmul(cid_rep, rid_rep.t())
        ref_acc_num = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size

        return kl_loss, hash_loss, quantization_loss, acc, ref_acc

class HashBERTSegmentDualEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTSegmentDualEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.hash_loss_matrix_scale = args['hash_loss_matrix_scale']
        self.kl_loss_scale = args['kl_loss_scale']
        self.dis_loss_scale = args['dis_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.seg_num = args['segment_num']
        self.seg_l = 768 // self.seg_num
        assert self.seg_l * self.seg_num == 768
        
        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        # gpu_ids = str(args['trainable_bert_layers'])
        # self.trainable_layers = [f'encoder.layer.{i}' for i in gpu_ids.split(',')]
        inpt_size = self.ctx_encoder.model.config.hidden_size
        self.ctx_hash_encoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inpt_size-self.seg_l, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.hash_code_size),
            ) for _ in range(self.seg_num)
        )
        self.can_hash_encoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inpt_size-self.seg_l, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.hash_code_size),
            ) for _ in range(self.seg_num)
        )
        self.ctx_hash_decoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hash_code_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, inpt_size-self.seg_l),
            ) for _ in range(self.seg_num)
        )
        self.can_hash_decoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hash_code_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, inpt_size-self.seg_l),
            ) for _ in range(self.seg_num)
        )

        self.kl_loss = torch.nn.MSELoss()
        self.dis_loss = torch.nn.KLDivLoss()

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
        segments_index = [list(range(i*self.seg_l, (i+1)*self.seg_l)) for i in range(self.seg_num)]
        hash_codes = []
        for idx in range(self.seg_num):
            selected_dims = list(chain(*[segments_index[j] for j in range(self.seg_num) if j != idx]))
            rid_rep_ = rid_rep[:, selected_dims]
            hash_code = torch.sign(self.can_hash_encoder(rid_rep_))
            hash_code = torch.where(
                hash_code == -1, 
                torch.zeros_like(hash_code), 
                hash_code,
            )
            hash_codes.append(hash_code)
        hash_codes = torch.cat(hash_codes, dim=-1)
        hash_codes = self.compact_binary_vectors(hash_codes)
        hash_codes = torch.from_numpy(hash_codes)
        return hash_codes
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, ids_mask)
        segments_index = [list(range(i*self.seg_l, (i+1)*self.seg_l)) for i in range(self.seg_num)]
        hash_codes = []
        for idx in range(self.seg_num):
            selected_dims = list(chain(*[segments_index[j] for j in range(self.seg_num) if j != idx]))
            cid_rep_ = cid_rep[:, selected_dims]
            hash_code = torch.sign(self.ctx_hash_encoder[idx](cid_rep))
            hash_code = torch.where(
                hash_code == -1, 
                torch.zeros_like(hash_code), 
                hash_code,
            )
            hash_codes.append(hash_code)
        hash_codes = torch.cat(hash_codes, dim=-1)
        hash_codes = self.compact_binary_vectors(hash_codes)
        return hash_codes

    @torch.no_grad()
    def predict(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = torch.ones_like(cid), batch['rids_mask']

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)

        segments_index = [list(range(i*self.seg_l, (i+1)*self.seg_l)) for i in range(self.seg_num)]
        ctx_hash_codes, can_hash_codes = [], []
        for idx in range(self.seg_num):
            selected_dims = list(chain(*[segments_index[j] for j in range(self.seg_num) if j != idx]))
            cid_rep_ = cid_rep[:, selected_dims]
            rid_rep_ = rid_rep[:, selected_dims]

            # [768]; [B, 768] -> [H]; [B, H]
            ctx_hash_code = torch.sign(self.ctx_hash_encoder[idx](cid_rep_))    # [1, Hash]
            can_hash_code = torch.sign(self.can_hash_encoder[idx](rid_rep_))    # [B, Hash]
            ctx_hash_codes.append(ctx_hash_code)
            can_hash_codes.append(can_hash_code)
        ctx_hash_codes = torch.cat(ctx_hash_codes, dim=-1)
        can_hash_codes = torch.cat(can_hash_codes, dim=-1)
        matrix = torch.matmul(ctx_hash_codes, can_hash_codes.t()).squeeze(0)    # [B]
        # minimal distance -> better performance 
        distance = -0.5 * (self.hash_code_size * self.seg_num - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        cid, rid = batch['ids'], batch['rids']
        cid_mask, rid_mask = batch['ids_mask'], batch['rids_mask']

        with torch.no_grad():
            cid_rep = self.ctx_encoder(cid, cid_mask)
            rid_rep = self.can_encoder(rid, rid_mask)
        
        batch_size = cid_rep.shape[0]
        r_batch_size = rid_rep.shape[0]

        kl_loss, hash_loss, quantization_loss = 0, 0, 0
        accs = []
        segments_index = [list(range(i*self.seg_l, (i+1)*self.seg_l)) for i in range(self.seg_num)]
        for idx in range(self.seg_num):

            selected_dims = list(chain(*[segments_index[j] for j in range(self.seg_num) if j != idx]))
            cid_rep_ = cid_rep[:, selected_dims]
            rid_rep_ = rid_rep[:, selected_dims]

            ctx_hash_code = self.ctx_hash_encoder[idx](cid_rep_)    # [B, H]
            can_hash_code = self.can_hash_encoder[idx](rid_rep_)    # [B*gray, H]
            ctx_hash_code_re = self.ctx_hash_decoder[idx](ctx_hash_code)    # [B, H]
            can_hash_code_re = self.can_hash_decoder[idx](can_hash_code)    # [B*gray, H]

            # ===== MSE Loss ===== #
            kl_loss_ = self.kl_loss(ctx_hash_code_re, cid_rep_) + self.kl_loss(can_hash_code_re, rid_rep_)
            kl_loss += kl_loss_ * self.kl_loss_scale

            # ===== calculate quantization loss ===== #
            ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
            quantization_loss += torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
            
            # ===== calculate hash loss ===== #
            matrix = torch.matmul(ctx_hash_code, can_hash_code.T)   # [B, B*H] similarity matrix
            mask = torch.zeros_like(matrix)
            mask[range(batch_size), range(r_batch_size)] = 1.
            label_matrix = self.hash_loss_matrix_scale * self.hash_code_size * mask
            hash_loss += torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale
            
            # ===== calculate hamming distance for accuracy ===== #
            matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
            hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
            acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
            acc = acc_num / batch_size
            accs.append(acc)

        # ===== ref acc ===== #
        dp = torch.matmul(cid_rep, rid_rep.t())
        ref_acc_num = (dp.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid))).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size

        acc = np.mean(accs)
        return kl_loss, hash_loss, quantization_loss, acc, ref_acc


