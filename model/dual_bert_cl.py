from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', p=0.2):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )

    def forward(self, ids, attn_mask, m=0):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        embd = self.head(embd[:, 0, :])
        return embd
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    

class BERTDualCLEncoder(nn.Module):

    def __init__(self, model='bert-base-chinese', head=5, p=0.2):
        super(BERTDualCLEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model, p=p)
        self.can_encoder = BertEmbedding(model=model, p=p)
        self.fusion = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 768),
        )

    def _encode(self, cid, rids, cid_mask, rids_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_reps = []
        for rid, rid_mask in zip(rids, rids_mask):
            rid_rep = self.can_encoder(rid, rid_mask)
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode_(cid.unsqueeze(0), rid, None, rid_mask)
        # cid_rep/rid_rep: [768], [B, 768]
        cid_rep = cid_rep.squeeze(0)    # [768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product

    def clloss(self, context, matrix1_, matrix2_):
        # context, matrix1/2: [B, 768]
        bsz = len(context)
        loss = 0
        for cid in context:
            cid = cid.repeat(bsz, 1)    # [B, 768]
            matrix1 = self.fusion(
                torch.cat([cid, matrix1_], dim=-1) 
            )    # [B, 768]
            matrix2 = self.fusion(
                torch.cat([cid, matrix2_], dim=-1) 
            )    # [B, 768]
            matrix1 = cid + matrix1_
            matrix2 = cid + matrix2_

            m1 = torch.matmul(matrix1, matrix1.t())    # [B, B]
            m2 = torch.matmul(matrix1, matrix2.t())    # [B, B]
            m3 = F.softmax(torch.cat([m1, m2], dim=1)[:, 1:], dim=1)    # [B, 2*B-1]
            loss += (m3[:, bsz-1] / m3.sum(dim=1)).mean()
        
            m4 = torch.matmul(matrix2, matrix2.t())    # [B, B]
            m5 = F.softmax(torch.cat([m4, m2], dim=1)[:, 1:], dim=1)   # [B, 2*B]
            loss += (m5[:, bsz-1] / m5.sum(dim=1)).mean()
        return loss
    
    def forward(self, cid, rids, cid_mask, rids_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_reps = self._encode(cid, rids, cid_mask, rids_mask)
        # ========== K matrixes =========== #
        # cid_rep: [B, 768]; rid_reps, rid_fusion: N*[B, 768]
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        acc, loss = 0, 0
        counter = 0
        for rid_rep in rid_reps:
            dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
            # calculate the loss
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss_ = (-loss_.sum(dim=1)).mean()
            loss += loss_

            # calculate the acc
            if counter == 0:
                acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
                acc = acc_num / batch_size
            counter += 1
        loss /= len(rid_reps)

        # CLLoss
        cl_loss = 0
        length = len(rid_reps)
        for i in range(length):
            for j in range(length):
                if j > i:
                    cl_loss += self.clloss(cid_rep, rid_reps[i], rid_reps[j])
        loss += cl_loss
        return loss, cl_loss, acc
    
    
class BERTDualCLEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None, head=10):
        super(BERTDualCLEncoderAgent, self).__init__()
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
            'head_num': head,
            'dropout': 0.2,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualCLEncoder(model=self.args['model'], head=self.args['head_num'], p=self.args['dropout'])
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
        total_loss, total_cl_loss, total_acc, batch_num = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            try:
                self.optimizer.zero_grad()
                cid, rid, cid_mask, rid_mask = batch
                loss, cl_loss, acc = self.model(cid, rid, cid_mask, rid_mask)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
                self.optimizer.step()
                self.scheduler.step()
    
                total_loss += loss.item()
                total_cl_loss += cl_loss.item()
                total_acc += acc
                batch_num += 1
                
                recoder.add_scalar(f'train-epoch-{idx_}/CLLoss', total_cl_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunCLLoss', cl_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                
                pbar.set_description(f'[!] loss: {round(total_loss/batch_num, 4)}|{round(total_cl_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    oom_t += 1
                    torch.cuda.empty_cache()
                    if oom_t > self.args['oom_times']:
                        raise Exception(f'[!] too much OOM errors')
                else:
                    raise Exception(str(exception))

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLLoss', total_cl_loss/batch_num, idx_)
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
