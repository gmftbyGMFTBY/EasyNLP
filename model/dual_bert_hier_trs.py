from .header import *
from .base import *
from .utils import *


class PositionEmbedding(nn.Module):

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
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        if model in ['bert-base-uncased']:
            # english corpus has three special tokens: __number__, __url__, __path__
            self.model.resize_token_embeddings(self.model.config.vocab_size + 3)

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        rest = embd[:, 0, :]    # [B, E]
        return rest
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)


class BERTDualHierarchicalTrsEncoder(nn.Module):

    '''try the transformers'''

    def __init__(self, model='bert-base-chinese', layer=3, nhead=6, dim_feedforward=512, dropout=0.1):
        super(BERTDualHierarchicalTrsEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            768,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(768)
        self.position_embd = PositionEmbedding(768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
        )
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
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
        '''resort and generate the order, context length mask'''
        # =========== reconstruct cid ========== #
        cid_rep = torch.split(cid_rep, cid_turn_length)
        # =========== padding =========== #
        max_turn_length = max([len(i) for i in cid_rep])
        cid_reps = []    # [B, S, E]
        cid_mask = []    # [B, S]
        for ctx in cid_rep:
            # mask, [S]
            m = torch.tensor([0] * len(ctx) + [1] * (max_turn_length - len(ctx))).to(torch.bool).half()
            cid_mask.append(m)
            if len(ctx) < max_turn_length:
                # support apex
                zero_tensor = torch.zeros(1, 768).half().cuda()
                padding = [zero_tensor] * (max_turn_length - len(ctx))
                ctx = torch.cat([ctx] + padding)    # append [S, E]
            cid_reps.append(ctx)
        cid_reps = torch.stack(cid_reps)
        cid_mask = cid_mask.cuda()
        return cid_reps, cid_mask    # [B, S, E], [B, S]

    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid_rep.shape[0]
        cid_rep, rid_rep = self._encode_(cid, rid, cid_mask, rid_mask)
        # [S, E], [10, E]
        cid_rep, cid_mask = self.reconstruct_tensor(cid_rep, cid_turn_length)
        ipdb.set_trace()
        
        cid_rep = cid_rep.unsqueeze(0)    # [1, S, E]
        cid_mask = cid_mask.unsqueeze(0)    # [1, S]

        cid_rep = self.position_embd(cid_rep)
        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding=cid_mask).permute(1, 0, 2)    # [1, S, E]
        cid_rep = self.proj(cid_rep)    # [B, 768]
        
        # step 1: obtain key
        # [B_c, B_r, E] x [B_c, E, S] -> [B_c, B_r, S]
        key = torch.bmm(rid_rep.repeat(1, 1, 1), cid_rep.permute(0, 2, 1))
        # step 2: mask the key, then softmax
        # [1, 10, S] * [1, 10, S] -> [1, 10, S]
        key = key * (~a).unsqueeze(1).repeat(1, batch_size, 1)
        key = F.softmax(key, dim=-1)
        # step 3: attention
        # [1, 10, S] x [1, S, E] -> [1, 10, E]
        cid_rep = torch.bmm(key, cid_rep)
        # step 4: dot production
        # [1, 10, E] enisum [1, 10, E] -> [1, 10]
        dot_product = torch.einsum('ijz, ijz -> ij', cid_rep, rid_rep.repeat(batch_size, 1, 1))
        dot_product = dot_product.squeeze(0)    # [10]
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
        cid_rep, cid_mask = self.reconstruct_tensor(cid_rep, cid_turn_length)
        ipdb.set_trace()

        cid_rep = self.position_embd(cid_rep)    # [B, S, E]
        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding=cid_mask).permute(1, 0, 2)    # [B, S, E]
        cid_rep = self.proj(cid_rep)

        # step 1: obtain key
        # [B_c, B_r, E] x [B_c, E, S] -> [B_c, B_r, S]
        key = torch.bmm(rid_rep.repeat(batch_size, 1, 1), cid_rep.permute(0, 2, 1))
        # step 2: mask the key, then softmax
        # [B_c, B_r, S] * [B_c, B_r, S] -> [B_c, B_r, S]
        key = key * (~a).unsqueeze(1).repeat(1, batch_size, 1)
        key = F.softmax(key, dim=-1)
        # step 3: attention
        # [B_c, B_r, S] x [B_c, S, E] -> [B_c, B_r, E]
        cid_rep = torch.bmm(key, cid_rep)
        # step 4: dot production
        # [B_c, B_r, E] enisum [B_c, B_r, E] -> [B_c, B_r]
        dot_product = torch.einsum('ijz, ijz -> ij', cid_rep, rid_rep.repeat(batch_size, 1, 1))

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
    
    
class BERTDualHierarchicalTrsEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualHierarchicalTrsEncoderAgent, self).__init__()
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
            'dropout': 0.1,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualHierarchicalTrsEncoder(
            model=self.args['model'], layer=self.args['gru_layer'], p=self.args['dropout']
        )
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
        self.show_parameters(self.args)
        
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping = batch
            loss, acc = self.model(cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
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
            assert batch_size == 10, f'[!] {batch_size} is not equal to 10'
            scores = self.model.module.predict(cid, rids, cid_turn_length, cids_mask, rids_mask).cpu().tolist()    # [B]
            
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
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map
