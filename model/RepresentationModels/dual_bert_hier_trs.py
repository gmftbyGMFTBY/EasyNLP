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

    def __init__(self, model='bert-base-chinese', nlayer=3, nhead=6, nhide=512, dropout=0.1):
        super(BERTDualHierarchicalTrsEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            768,
            nhead=nhead,
            dim_feedforward=nhide,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(768)
        # max context utterances is 512
        self.position_embd = nn.Embedding(512, 768)
        self.speaker_embd = nn.Embedding(2, 768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            nlayer,
            encoder_norm,
        )
        # self.gru = nn.GRU(
        #     768, 768, nlayer, batch_first=True,
        #     dropout=0 if nlayer == 1 else dropout,
        # )
        # self.proj = nn.Sequential(
        #      nn.Linear(nlayer*768, 768),
        #      nn.Dropout(p=dropout),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        #  )

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
            m = torch.tensor([0] * len(ctx) + [1] * (max_turn_length - len(ctx))).to(torch.bool)
            cid_mask.append(m)
            if len(ctx) < max_turn_length:
                # support apex
                zero_tensor = torch.zeros(1, 768).half().cuda()
                padding = [zero_tensor] * (max_turn_length - len(ctx))
                ctx = torch.cat([ctx] + padding)    # append [S, E]
            cid_reps.append(ctx)
        pos_index = torch.arange(max_turn_length).repeat(len(cid_rep), 1).cuda()    # [B, S]
        cid_reps = torch.stack(cid_reps)
        cid_mask = torch.stack(cid_mask).cuda()
        spk_index = torch.ones(len(cid_rep), max_turn_length).cuda()    # [B, S]
        spk_index[:, ::2] = 0
        spk_index = spk_index.to(torch.long)
        return cid_reps, cid_mask, pos_index, spk_index  # [B, S, E], [B, S], [B, S]
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid.shape[0]
        
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # [S, E], [10, E]
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)
        
        pos_embd = self.position_embd(pos_index)
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [1, S, E]

        cid_rep += cid_rep_base

        cid_rep = cid_rep[:, cid_turn_length-1, :]    # [1, E]

        # _, cid_rep = self.gru(cid_rep)
        #  cid_rep = cid_rep.permute(1, 0, 2)
        #  id_rep = cid_rep.reshape(cid_rep.shape[0], -1)
        # cid_rep = self.proj(cid_rep)
        # cid_rep += cid_rep_jump
        
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()    # [10] 
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
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)

        # Transformer Encoder
        pos_embd = self.position_embd(pos_index)    # [B, S, E]
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [B, S, E]

        cid_rep += cid_rep_base

        last_utterance = []
        for i in range(len(cid_turn_length)):
            c = cid_rep[i]
            p = cid_turn_length[i]
            last_utterance.append(c[p-1, :])
        cid_rep = torch.stack(last_utterance)    # [B_c, E]

        #  cid_rep = nn.utils.rnn.pack_padded_sequence(cid_rep, cid_turn_length, batch_first=True, enforce_sorted=False)
        # _, cid_rep = self.gru(cid_rep)
        # cid_rep = cid_rep.permute(1, 0, 2)
        #  cid_rep = cid_rep.reshape(cid_rep.shape[0], -1)
        # cid_rep = self.proj(cid_rep)
        #  cid_rep += cid_rep_jump

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # c-r
        loss_1 = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_1.sum(dim=1)).mean()
        # r-c
        # loss_2 = F.log_softmax(dot_product, dim=0) * mask
        #  loss += (-loss_2.sum(dim=0)).mean()
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
            'local_rank': local_rank,
            'warmup_steps': warmup_step,
            'total_step': total_step,
            'max_len': 256,
            'dataset': dataset_name,
            'pretrained_model_path': pretrained_model_path,
            'nhead': 6,
            'nhide': 512,
            'nlayer': 2,
            'dropout': 0.3,
            'amp_level': 'O2',
            'test_interval': 0.05
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0

        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualHierarchicalTrsEncoder(
            model=self.args['model'], 
            nlayer=self.args['nlayer'], 
            nhide=self.args['nhide'], 
            nhead=self.args['nhead'], 
            dropout=self.args['dropout']
        )
        if pretrained_model_path:
            if run_mode == 'train-post':
                self.load_bert_model(pretrained_model_path)
            elif run_mode == 'train-dual-post':
                self.load_dual_bert_model(pretrained_model_path)
            else:
                raise Exception(f'[!] Unknown run mode: {run_mode}')
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        if run_mode in ['train', 'train-post', 'train-dual-post']:
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                opt_level=self.args['amp_level']
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

    def load_dual_bert_model(self, path):
        # collect
        def collect_params(name):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith(name):
                    k = k.replace(f'{name}.', '')
                    new_state_dict[k] = v
            return new_state_dict
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        ctx_ckpt, can_ckpt = collect_params('ctx_encoder'), collect_params('can_encoder')
        self.model.ctx_encoder.load_state_dict(ctx_ckpt)
        self.model.can_encoder.load_state_dict(can_ckpt)
        print(f'[!] load the dual-bert-post pretrained BERT model from {path}')
        
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
            loss, acc = self.model(
                cid, rid, cid_turn_length, 
                cid_mask, rid_mask, recover_mapping
            )
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                # test in the training loop
                index = self.test_step_counter
                (r10_1, r10_2, r10_5), avg_mrr, avg_p1, avg_map = self.test_model()
                self.model.train()
                recoder.add_scalar(f'train-test/R10@1', r10_1, index)
                recoder.add_scalar(f'train-test/R10@2', r10_2, index)
                recoder.add_scalar(f'train-test/R10@5', r10_5, index)
                recoder.add_scalar(f'train-test/MRR', avg_mrr, index)
                recoder.add_scalar(f'train-test/P@1', avg_p1, index)
                recoder.add_scalar(f'train-test/MAP', avg_map, index)
                self.test_step_counter += 1

            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        pbar = tqdm(self.test_iter)
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
