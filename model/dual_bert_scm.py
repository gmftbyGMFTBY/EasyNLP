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
    
    def __init__(self, nhead, nlayer, ndim, dropout=0.1, model='bert-base-chinese'):
        super(BERTDualCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            768, nhead=nhead, dim_feedforward=ndim, dropout=dropout        
        )
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer, nlayer, encoder_norm
        )
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768*2, 768),
        )
        self.pos_embd = nn.Embedding(512, 768)
        self.seg_embd = nn.Embedding(2, 768)
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def _get_input_label(self, cid_rep, rid_rep, mode='train'):
        '''cid_rep: [B, E]; rid_rep: [B, E];
        return the hidden state and the label'''
        ctx_batch_size, res_batch_size = len(cid_rep), len(rid_rep)
        # random shuffle and the get the label
        rid_rep = rid_rep.repeat(ctx_batch_size, 1, 1)    # [B_c, B_r, E]
        label, rid_reps = [], []

        # pos embedding
        pos_index = torch.arange(1+res_batch_size).cuda()
        pos_embd = self.pos_embd(pos_index)    # [B_r+1, E] 
        pos_embd = pos_embd.unsqueeze(0).repeat(ctx_batch_size, 1, 1)    # [B_c, B_r+1, E]

        # segment embedding
        seg_index = torch.LongTensor([1] + [0] * res_batch_size).cuda()
        seg_embd = self.seg_embd(seg_index)
        seg_embd = seg_embd.unsqueeze(0).repeat(ctx_batch_size, 1, 1)    # [B_c, B_r+1, E]

        for i in range(ctx_batch_size):
            item = rid_rep[i]    # [B_r, E]
            index = list(range(res_batch_size))

            if mode == 'train':
                random.shuffle(index)
                # append label
                l = torch.LongTensor([0] * res_batch_size)
                l[index.index(i)] = 1
                label.append(l)

            # update rid rep
            item = [item[i] for i in index]
            item = torch.stack(item)
            rid_reps.append(item)
        rid_reps = torch.stack(rid_reps)    # [B_c, B_r, E]
        ipt = torch.cat([cid_rep.unsqueeze(1), rid_reps], dim=1)    # [B_c, B_r+1, E]
        ipt += pos_embd
        ipt += seg_embd
        
        if mode == 'train':
            label = torch.stack(label).cuda()    # [B_c, B_r]
            # [B_c, B_r+1, E]; [B_c, B_r]
            return ipt, label
        else:
            return ipt
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        # cid_rep: [1, E]; rid_rep: [S, E]
        batch_size = rid.shape[0]
        cid = cid.unsqueeze(0)
        cid_rep, rid_rep = self._encode(cid, rid, None, rid_mask)    # [B_c, E], [B_r, E]
        ipt = self._get_input_label(cid_rep, rid_rep, mode='test')    # [B_c, B_r, E]
        
        ipt = ipt.permute(1, 0, 2)    # [B_r+1, B_c, E]
        opt = self.trs_encoder(ipt)    # [B_r+1, B_c, E]
        opt = self.proj2(torch.cat([ipt, opt], dim=-1))
        opt = self.proj(opt).permute(1, 0, 2)    # [B_c, B_r+1, E]
        opt = opt[0, 1:, :]    # [B_r, E], ignore the context information

        dot_product = torch.matmul(cid_rep, opt.t())    # [B_c, B_r]
        return dot_product.squeeze(0)    # [B_r]
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, E]
        ipt, label = self._get_input_label(cid_rep, rid_rep)

        ipt = ipt.permute(1, 0, 2)    # [B_r+1, B_c, E]
        opt = self.trs_encoder(ipt)    # [B_r+1, B_c, E]
        opt = self.proj2(torch.cat([ipt, opt], dim=-1))
        opt = self.proj(opt).permute(1, 0, 2)    # [B_c, B_r+1, E]
        opt = opt[:, 1:, :]    # [B_c, B_r, E], ignore the context information

        # [B_c, B_r]: [B_c, 1, E] x [B_c, B_r, E]
        dot_product = torch.bmm(
            cid_rep.unsqueeze(1),
            opt.permute(0, 2, 1)
        ).squeeze(1)    # [B_c, B_r]

        loss = F.log_softmax(dot_product, dim=-1) * label
        loss = (-loss.sum(dim=1)).mean()
        
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == label.nonzero()[:, 1]).sum().item()
        acc = 100 * acc_num / batch_size
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
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': warmup_step,
            'total_step': total_step,
            'nhead': 6,
            'nlayer': 4,
            'ndim': 512,
            'dropout': 0.1,
            'test_interval': 0.05,
            'model': pretrained_model,
            'pretrained_model_path': pretrained_model_path,
            'dataset': dataset_name,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualCompEncoder(
            self.args['nhead'], 
            self.args['nlayer'],
            self.args['ndim'],
            dropout=self.args['dropout'],
            model=self.args['model'],
        )
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode in ['train', 'train-post', 'train-dual-post']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(),
                lr=self.args['lr'],
            )
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

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 2)}|{round(total_acc/batch_num, 2)}')
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
            total_examples += 1
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map
