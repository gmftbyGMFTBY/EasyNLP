from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', p=0.2):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 768),
        )

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return self.head(embd[:, 0, :])
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    

class BERTDualJSDEncoder(nn.Module):
    
    # NOTE
    def __init__(self, model='bert-base-chinese', p=0.2, head=5, alpha=10):
        super(BERTDualJSDEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model, p=p)
        self.can_encoder = BertEmbedding(model=model, p=p)

        # NOTE
        self.ctx_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(768, 768)
            ) for _ in range(head)
        ])
        self.res_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(768, 768)
            ) for _ in range(head)
        ])
        self.head_num = head
        self.alpha = alpha

    def _encode(self, cid, rid, cid_mask, rid_mask):
        # NOTE
        cid_rep = self.ctx_encoder(cid, cid_mask)
        cid_reps = [self.ctx_head[i](cid_rep) for i in range(self.head_num)]
        rid_rep = self.can_encoder(rid, rid_mask)
        rid_reps = [self.res_head[i](rid_rep) for i in range(self.head_num)]
        return cid_reps, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_reps, rid_reps = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        dot_product = []
        for cid, rid in zip(cid_reps, rid_reps):
            cid = cid.squeeze(0)    # [768]
            # cid_rep/rid_rep: [768], [B, 768]
            dot_product.append(torch.matmul(cid, rid.t()))  # [B]
        dot_product = torch.stack(dot_product)
        dot_product = dot_product.max(dim=0)[0]
        return dot_product

    def jsd(self, p, q):
        m = F.softmax(0.5 * (p + q), dim=1)
        d = 0.5 * F.kl_div(F.log_softmax(p, dim=1), m, reduction='batchmean') + 0.5 * F.kl_div(F.log_softmax(q, dim=1), m, reduction='batchmean')
        return d
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_reps, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, 768]
        loss, acc = 0, 0
        for cid_rep, rid_rep in zip(cid_reps, rid_reps):
            dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
            mask = torch.eye(batch_size).cuda().half()    # [B, B]
            # calculate accuracy
            acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
            acc_ = acc_num / batch_size
            acc += acc_
            # calculate the loss
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
        acc /= self.head_num
        d_loss, counter = 0, 0
        # JS-Divergence Diveristy
        for i in range(self.head_num):
            for j in range(self.head_num):
                if i != j:
                    d_loss -= self.alpha * self.jsd(cid_reps[i], cid_reps[j])
                    d_loss -= self.alpha * self.jsd(rid_reps[i], rid_reps[j])
                    counter += 2
        # d_loss /= counter
        loss += d_loss
        return loss, -d_loss, acc
    
    
class BERTDualJSDEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualJSDEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,     # dot production: 5e-5, cosine similairty: 1e-4
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
            'dropout': 0.2,
            'head_num': 5,
            'alpha': 1,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualJSDEncoder(model=self.args['model'], p=self.args['dropout'], head=self.args['head_num'], alpha=self.args['alpha'])
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
        '''ADD OOM ASSERTION'''
        self.model.train()
        total_loss, total_acc, batch_num, total_jsd_loss = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            try:
                self.optimizer.zero_grad()
                cid, rid, cid_mask, rid_mask = batch
                loss, jsd_loss, acc = self.model(cid, rid, cid_mask, rid_mask)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
                self.optimizer.step()
                self.scheduler.step()
    
                total_loss += loss.item()
                total_jsd_loss += jsd_loss.item()
                total_acc += acc
                batch_num += 1
                
                recoder.add_scalar(f'train-epoch-{idx_}/JSDLoss', total_jsd_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunJSDLoss', jsd_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                
                pbar.set_description(f'[!] OOM: {oom_t}; loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; jsd_loss: {round(jsd_loss.item(), 4)}|{round(total_jsd_loss/batch_num, 4)} acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    oom_t += 1
                    torch.cuda.empty_cache()
                    if oom_t > self.args['oom_times']:
                        raise Exception(f'[!] too much OOM errors')
                else:
                    raise Exception(str(exception))

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/JSDLoss', total_jsd_loss/batch_num, idx)
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
            cid, rids, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} is not equal to 10'
            scores = self.model.predict(cid, rids, rids_mask).cpu().tolist()    # [B]

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
            vec = self.model.module.get_cand(ids, mask).cpu()    # [B, H]
            matrix.append(vec)
            corpus.extend(text)
        matrix = torch.cat(matrix, dim=0).numpy()    # [Size, H]
        assert len(matrix) == len(corpus)

        # context response
        pbar = tqdm(test_iter)
        for batch in pbar:
            ids, ids_mask, ctx_text, res_text, order = batch
            vec = self.model.module.get_ctx(ids, ids_mask).cpu()
            queries.append(vec)
            q_text.extend(ctx_text)
            q_text_r.extend(res_text)
            q_order.extend(order)
        queries = torch.cat(queries, dim=0).numpy()
        torch.save(
            (queries, q_text, q_text_r, q_order, matrix, corpus), 
            f'data/{self.args["dataset"]}/inference_{self.args["local_rank"]}.pt'
        )
