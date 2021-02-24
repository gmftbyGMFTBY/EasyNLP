from .header import *
from .base import *
from .utils import *


class PositionEmbedding(nn.Module):

    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.5, max_len=512):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BertEmbedding(nn.Module):
    
    def __init__(self, m=0, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.m = m

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        if self.m == 0:
            rest = embd[:, 0, :]
        else:
            rest = []
            cid_ls = [min(len(item.nonzero().squeeze()), self.m) for item in attn_mask]
            for idx in range(len(embd)):
                rest.append(embd[idx][:cid_ls[idx], :])
        # [B, E]/B*[M, E]
        return rest
    
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f'[!] load pretrained BERT model from {path}')


class BERTDualHierarchicalMultiEncoder(nn.Module):
    
    '''switch transformer to one layer GRU'''

    def __init__(self, model='bert-base-chinese', layer=2, m=5, p=0.1, nhead=6, dim_ffd=512, num_encoder_layers=4):
        super(BERTDualHierarchicalMultiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, model=model)
        self.can_encoder = BertEmbedding(m=0, model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            768,
            nhead=nhead,
            dim_feedforward=dim_ffd,
            dropout=p
        )
        self.position_embd = PositionEmbedding(768)
        encoder_norm = nn.LayerNorm(768)
        self.ctx_trs = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
        )

        self.m = m
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
    def _encode(self, cids, rid, cids_mask, rid_mask, recover_mapping, cid_turn_length):
        '''resort'''
        cid_reps = []    # k*[B, M, E]
        for cid, cid_mask in zip(cids, cids_mask):
            cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, M, E]
            cid_reps.extend(cid_rep)
        # recover
        cid_reps = [cid_reps[recover_mapping[idx]] for idx in range(len(cid_reps))]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, E]
        return cid_reps, rid_rep

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # B*[M, E]
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    def reconstruct_tensor_(self, cid_rep):
        '''resort and generate the order'''
        # =========== reconstruct cid ========== #
        # cid_rep: [L, M, E]; L = B*k
        ctx = cid_rep[:]
        # ==========
        p_, pos = [], 0
        for i in ctx:
            p_.extend([pos] * i.shape[0])
            pos += 1
        pos = p_
        # ==========
        ctx = torch.cat(ctx)    # [L, E]
        # =========== padding =========== #
        cid_reps_, pos_index_ = [], []    # [B, S, E]; [B, S]
        # 512 is the max_length
        if len(ctx) > 512:
            ctx = ctx[-512:, :]
            pos = pos[-512:]
        pos = torch.LongTensor(pos).cuda()
        assert pos.shape[0] == ctx.shape[0]
        # mask: [B, L], True ignored
        cid_reps = ctx.unsqueeze(0)    # [1, L, E]
        pos_index = pos.unsqueeze(0)    # [1, L]
        return cid_reps, pos_index    # [1, L, E]; [1, L]

    def reconstruct_tensor(self, cid_rep, cid_turn_length):
        '''resort and generate the order'''
        # =========== reconstruct cid ========== #
        # cid_rep: [L, M, E]; L = B*k
        cid_reps, index, turn_length_collector, pos_index = [], 0, [], []
        # pos_index: [B, S]
        for turn_length in cid_turn_length:
            ctx = cid_rep[index:index+turn_length]
            # ==========
            p_, pos = [], 0
            for i in ctx:
                p_.extend([pos] * i.shape[0])
                pos += 1
            pos_index.append(p_)
            # ==========
            ctx = torch.cat(ctx)    # [L, E]
            cid_reps.append(ctx)
            turn_length_collector.append(len(ctx))
            index += turn_length
        # cid_turn_length = B*k
        # =========== padding =========== #
        cid_reps_, pos_index_ = [], []    # [B, S, E]; [B, S]
        max_turn_length = max(turn_length_collector)
        for ctx, pos in zip(cid_reps, pos_index):
            # ctx: [L, E]
            if max_turn_length > 512:
                # 512 is the max_length
                if len(ctx) > 512:
                    ctx = ctx[-512:, :]
                    pos = pos[-512:]
                else:
                    zero_tensor = torch.zeros(1, 768).cuda()
                    padding = [zero_tensor] * (512 - len(ctx))
                    ctx = torch.cat([ctx] + padding)    # [L, E]
                    pos += [pos[-1] + 1] * (512 - len(pos))
            else:
                if len(ctx) < max_turn_length:
                    # support apex
                    zero_tensor = torch.zeros(1, 768).half().cuda()
                    # zero_tensor = torch.zeros(1, 768).cuda()
                    padding = [zero_tensor] * (max_turn_length - len(ctx))
                    ctx = torch.cat([ctx] + padding)    # [L, E]
                    pos += [pos[-1] + 1] * (max_turn_length - len(pos))
            pos = torch.LongTensor(pos).cuda()
            assert pos.shape[0] == ctx.shape[0]
            cid_reps_.append(ctx)
            pos_index_.append(pos)
        # mask: [B, L], True ignored
        cid_reps = torch.stack(cid_reps_)    # [B, L, E]
        ctx_mask = torch.zeros(cid_reps.shape[0], cid_reps.shape[1], dtype=torch.bool).cuda()
        pos_index = torch.stack(pos_index_)    # [B, L]
        for i in range(len(turn_length_collector)):
            ctx_mask[i, turn_length_collector[i]:] = True
        return cid_reps, ctx_mask, pos_index    # [B, L, E]; [B, L]

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode_(cid, rid, cid_mask, rid_mask)    # B*[M, E]
        cid_rep, pos_index = self.reconstruct_tensor_(cid_rep)
        cid_rep = cid_rep.permute(1, 0, 2)     # [L, 1, E]

        # hierarchical encode
        # cid_rep += self.position_embd(pos_index)
        cid_rep = self.position_embd(cid_rep)
        cid_rep = self.ctx_trs(cid_rep)    # [L, 1, E]
        cid_rep = cid_rep.mean(dim=0).squeeze()    # [E], pooling
        cid_rep = self.proj(cid_rep)    # [768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping):
        '''parameters:
        cid: [B_k, S]; B_k = B * \sum_{k=1}^B S_k
        cid_mask: [B_k, S]
        rid: [B, S];
        rid_mask: [B_k, S];
        cid_turn_length: [B]'''
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, recover_mapping, cid_turn_length)
        cid_rep, cid_mask, pos_index = self.reconstruct_tensor(cid_rep, cid_turn_length)
        cid_rep = cid_rep.permute(1, 0, 2)     # [L, B, E]

        # hierarchical encode
        # cid_rep += self.position_embd(pos_index)
        cid_rep = self.position_embd(cid_rep)
        cid_rep = self.ctx_trs(cid_rep, src_key_padding_mask=cid_mask)    # [L, B, E]
        cid_rep = cid_rep.mean(dim=0)    # [B, E], pooling
        cid_rep = self.proj(cid_rep)    # [B, 768]

        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        # mask = torch.eye(batch_size).cuda()    # [B, B]
        # mask = torch.eye(batch_size).cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    

class BERTDualHierarchicalEncoder(nn.Module):

    '''try the transformers'''

    def __init__(self, model='bert-base-chinese', layer=2, p=0.1):
        super(BERTDualHierarchicalEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        self.ctx_gru = nn.GRU(
            768, 768, layer, batch_first=True,
            dropout=0 if layer == 1 else p,
        )
        self.layer = layer
        self.proj = nn.Sequential(
            nn.Linear(layer*768, 768),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
    def _encode(self, cids, rid, cids_mask, rid_mask, recover_mapping):
        '''resort'''
        cid_reps = []
        for cid, cid_mask in zip(cids, cids_mask):
            cid_rep = self.ctx_encoder(cid, cid_mask)
            cid_reps.append(cid_rep)
        cid_reps = torch.cat(cid_reps)    # [B, E]
        # recover
        cid_reps = [cid_reps[recover_mapping[idx]] for idx in range(len(cid_reps))]
        cid_rep = torch.stack(cid_reps)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def reconstruct_tensor(self, cid_rep, cid_turn_length):
        '''resort and generate the order'''
        # =========== reconstruct cid ========== #
        cid_rep = torch.split(cid_rep, cid_turn_length)
        # =========== padding =========== #
        max_turn_length = max([len(i) for i in cid_rep])
        cid_reps = []    # [B, S, E]
        for ctx in cid_rep:
            if len(ctx) < max_turn_length:
                # support apex
                zero_tensor = torch.zeros(1, 768).half().cuda()
                padding = [zero_tensor] * (max_turn_length - len(ctx))
                ctx = torch.cat([ctx] + padding)
            cid_reps.append(ctx)
        cid_reps = torch.stack(cid_reps)
        return cid_reps    # [B, S, E]

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode_(cid, rid, cid_mask, rid_mask)    # [k, E]
        cid_rep = self.reconstruct_tensor(cid_rep, cid_turn_length)
        _, cid_rep = self.ctx_gru(cid_rep)    # [1, B, 768]
        cid_rep = cid_rep.permute(1, 0, 2)    # [B, layer, 768]
        cid_rep = cid_rep.reshape(cid_rep.shape[0], -1)    # [B, layer*768]
        cid_rep = self.proj(cid_rep)    # [B, 768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping):
        '''parameters:
        cid: [B_k, S]; B_k = B * \sum_{k=1}^B S_k
        cid_mask: [B_k, S]
        rid: [B, S];
        rid_mask: [B_k, S];
        cid_turn_length: [B]'''
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, recover_mapping)
        cid_rep = self.reconstruct_tensor(cid_rep, cid_turn_length)

        cid_rep = nn.utils.rnn.pack_padded_sequence(cid_rep, cid_turn_length, batch_first=True, enforce_sorted=False)
        _, cid_rep = self.ctx_gru(cid_rep)    # [layer, B, 768]
        cid_rep = cid_rep.permute(1, 0, 2)    # [B, layer, 768]
        cid_rep = cid_rep.reshape(cid_rep.shape[0], -1)    # [B, layer*768]
        cid_rep = self.proj(cid_rep)    # [B, 768]

        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        # mask = torch.eye(batch_size).cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
    
class BERTDualHierarchicalEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualHierarchicalEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'model': pretrained_model,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': warmup_step,
            'total_step': total_step,
            'max_len': 256,
            'dataset': dataset_name,
            'pretrained_model_path': pretrained_model_path,
            'oom_times': 10,
            'gru_layer': 2,
            'm': 5,
            'dropout': 0.1,
            'num_encoder_layers': 4,
            'dim_ffd': 512,
            'nhead': 6,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        # self.model = BERTDualHierarchicalMultiEncoder(model=self.args['model'], num_encoder_layers=self.args['num_encoder_layers'], dim_ffd=self.args['dim_ffd'], nhead=self.args['nhead'], m=self.args['m'])
        self.model = BERTDualHierarchicalEncoder(model=self.args['model'], layer=self.args['gru_layer'], p=self.args['dropout'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        elif run_mode == 'inference':
            # self.model = amp.initialize(
            #     self.model, 
            #     opt_level=self.args['amp_level'],
            # )
            # self.model = nn.parallel.DistributedDataParallel(
            #     self.model, device_ids=[local_rank], output_device=local_rank,
            #     find_unused_parameters=True,
            # )
            pass
        self.show_parameters(self.args)
        
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''ADD OOM ASSERTION'''
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            try:
                self.optimizer.zero_grad()
                cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping = batch
                loss, acc = self.model(cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
                # clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()
                self.scheduler.step()
    
                total_loss += loss.item()
                total_acc += acc
                batch_num += 1
                
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                
                pbar.set_description(f'[!] OOM: {oom_t}; loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    oom_t += 1
                    torch.cuda.empty_cache()
                    if oom_t > self.args['oom_times']:
                        raise Exception(f'[!] too much OOM errors')
                else:
                    raise Exception(str(exception))

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, recoder=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            cid, rids, cid_turn_length, cids_mask, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} isnot equal to 10'
            scores = self.model.predict(cid, rids, cid_turn_length, cids_mask, rids_mask).cpu().tolist()    # [B]
            
            rank_by_pred, pos_index, stack_scores = \
          calculate_candidates_ranking(
                np.array(scores), 
                np.array(label.cpu().tolist()),
                10)
            num_correct = logits_recall_at_k(pos_index, k_list)
            if self.args['dataset'] in ["douban"]:
                total_prec_at_one += precision_at_one(rank_by_pred)
                total_map += mean_average_precision(pos_index)
                for pred in rank_by_pred:
                    if sum(pred) == 0:
                        total_examples -= 1
            total_mrr += logits_mrr(pos_index)
            total_correct = np.add(total_correct, num_correct)
            total_examples += math.ceil(label.size()[0] / 10)
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")

    @torch.no_grad()
    def inference(self, inf_iter, test_iter):
        self.model.eval()
        pbar = tqdm(inf_iter)
        matrix, corpus, queries, q_text, q_order, q_text_r = [], [], [], [], [], []
        for batch in pbar:
            ids, mask, text = batch
            vec = self.model.get_cand(ids, mask).cpu()    # [B, H]
            matrix.append(vec)
            corpus.extend(text)
        matrix = torch.cat(matrix, dim=0).numpy()    # [Size, H]
        assert len(matrix) == len(corpus)

        # context response
        pbar = tqdm(test_iter)
        for batch in pbar:
            ids, ids_mask, ctx_text, res_text, order = batch
            vec = self.model.get_ctx(ids, ids_mask).cpu()
            queries.append(vec)
            q_text.extend(ctx_text)
            q_text_r.extend(res_text)
            q_order.extend(order)
        queries = torch.cat(queries, dim=0).numpy()
        torch.save(
            (queries, q_text, q_text_r, q_order, matrix, corpus), 
            f'data/{self.args["dataset"]}/inference_{self.args["local_rank"]}.pt'
        )
