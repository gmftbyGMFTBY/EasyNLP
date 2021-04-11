from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        if model in ['bert-base-uncased']:
            self.model.resize_token_embeddings(self.model.config.vocab_size+3)

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        rest = embd[:, 0, :]
        return rest

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    
class BERTDualCompEncoder(nn.Module):
    
    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, model='bert-base-chinese'):
        super(BERTDualCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            768, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        
        self.proj1 = nn.Linear(768*2, 768)
        self.gate = nn.Linear(768*3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        # cid_rep: [1, E]; rid_rep: [S, E]
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [E]
        cross_rep = torch.cat(
            [
                cid_rep.unsqueeze(0).expand(batch_size, -1), 
                rid_rep,
            ], 
            dim=1,
        )    # [S, 2*E]
        
        cross_rep = self.dropout(
            torch.tanh(
                self.trs_encoder(
                    torch.tanh(
                        self.proj1(cross_rep).unsqueeze(1)
                    )
                )
            ).squeeze(1)
        )    # [S, E]
        
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        rid_rep,    # [S, E]
                        cid_rep.unsqueeze(0).expand(batch_size, -1),    # [S, E]
                        cross_rep,    # [S, E]
                    ],
                    dim=-1,
                )
            )
        )    # [S, E]
        # cross_rep: [S, E]
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        # cid: [E]; cross_rep: [S, E]
        dot_product = torch.matmul(cid_rep, cross_rep.t())    # [S]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, E]
        
        # cross attention for all the candidates
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)    # [S, E]
            cross_rep.append(
                torch.cat([cid_rep_, rid_rep], dim=-1)
            )    # [S, E*2]
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)    # [B, S, 2*E] -> [S, B, E*2]
        cross_rep = self.dropout(
            torch.tanh(
                self.trs_encoder(
                    torch.tanh(self.proj1(cross_rep))
                )
            ).permute(1, 0, 2)
        )    # [B, S, E]
        
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        rid_rep.unsqueeze(0).expand(batch_size, -1, -1), 
                        cid_rep.unsqueeze(1).expand(-1, batch_size, -1),
                        cross_rep,
                    ], 
                    dim=-1
                )
            )
        )    # [B, S, E]
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)    # [B, S, E]
        
        # reconstruct rid_rep
        cid_rep = cid_rep.unsqueeze(1)    # [B, 1, E]
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)    # [B, S]
        # use half for supporting the apex
        mask = torch.eye(batch_size).half().cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class BERTDualCompEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualCompEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'lr_': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'samples': 10,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': warmup_step,
            'total_step': total_step,
            'num_encoder_layers': 2,
            'dim_feedforward': 512,
            'nhead': 6,
            'dropout': 0.1,
            'max_len': 256,
            'test_interval': 0.05,
            'model': pretrained_model,
            'pretrained_model_path': pretrained_model_path,
            'max_len': 256,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualCompEncoder(
            self.args['nhead'], 
            self.args['dim_feedforward'], 
            self.args['num_encoder_layers'], 
            dropout=self.args['dropout'],
            model=self.args['model']
        )
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode in ['train', 'train-post', 'train-dual-post']:
            self.optimizer = transformers.AdamW(
                [
                    {
                        'params': self.model.ctx_encoder.parameters(),
                    },
                    {
                        'params': self.model.can_encoder.parameters(),
                    },
                    {
                        'params': self.model.trs_encoder.parameters(), 
                        'lr': self.args['lr_'],
                    },
                    {
                        'params': self.model.proj1.parameters(), 
                        'lr': self.args['lr_'],
                    },
                    {
                        'params': self.model.gate.parameters(), 
                        'lr': self.args['lr_'],
                    }
                ], 
                lr=self.args['lr'],
            )
            print(f'[!] set the different learning ratios for comparsion module')
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_steps'], 
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
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            loss, acc = self.model(cid, rid, cid_mask, rid_mask)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
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
            cid, rids, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} is not equal to 10'
            scores = self.model.module.predict(cid, rids, rids_mask).cpu().tolist()    # [B]

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
            # total_examples += math.ceil(label.size()[0] / 10)
            total_examples += 1
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        avg_time = np.mean(self.model.module.test_time_cost)
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")
        print(f"Avg Time Cost: {round(1000*avg_time, 5)} ms")
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map
