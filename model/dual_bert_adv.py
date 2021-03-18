from .header import *
from .base import *
from .utils import *


'''dual bert with adversarial negative samples. 2021-CVPR: AdCo'''


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, ids, attn_mask):
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


class Adv(nn.Module):

    def __init__(self, K):
        super(Adv, self).__init__()
        self.K = K
        self.adversaries = nn.Parameter(torch.FloatTensor(self.K, 768))

    def forward(self, x):
        adv_rep = self.adversaries    # [K, 768]
        return adv_rep
    

class BERTDualAdvEncoder(nn.Module):
    
    def __init__(self, model='bert-base-chinese', K=65536):
        super(BERTDualAdvEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, 768]
        rid_rep = self.can_encoder(rid, rid_mask)    # [B, 768]   
        return cid_rep, rid_rep

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
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask, adv_rep):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep/adv_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        dot_product_ = torch.matmul(cid_rep, adv_rep.t())    # [B, K]
        dot_product = torch.cat([dot_product, dot_product_], dim=1)    # [B, B+K]
        # use half for supporting the apex
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        zeros = torch.zeros_like(dot_product_).cuda().half()
        mask = torch.cat([mask, zeros], dim=1)    # [B, 2*B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc, cid_rep.detach(), rid_rep.detach(), mask 

    
class BERTDualAdvEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualAdvEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,     # dot production: 5e-5, cosine similairty: 1e-4
            'adv_lr': 5e-5,     # dot production: 5e-5, cosine similairty: 1e-4
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
            'K': 65536,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualAdvEncoder(model=self.args['model'])
        self.adv = Adv(self.args['K'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
            self.adv.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.optimizer_adv = transformers.AdamW(
            self.adv.parameters(), 
            lr=self.args['adv_lr'],
        )
        if run_mode == 'train':
            [self.model, self.adv], [self.optimizer, self.optimizer_adv] = amp.initialize(
                [self.model, self.adv], 
                [self.optimizer, self.optimizer_adv],
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=total_step,
            )
            self.scheduler_adv = transformers.get_linear_schedule_with_warmup(
                self.optimizer_adv, 
                num_warmup_steps=warmup_step, 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
            self.adv = nn.parallel.DistributedDataParallel(
                self.adv, device_ids=[local_rank], output_device=local_rank,
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
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            adv_rep = self.adv(None)
            loss, acc, cid_rep, rid_rep, mask = self.model(cid, rid, cid_mask, rid_mask, adv_rep.detach())
            # loss for adv encoder, maximun the loss
            dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
            dot_product_ = torch.matmul(cid_rep, adv_rep.t())    # [B, K]
            dot_product = torch.cat([dot_product, dot_product_], dim=1)    # [B, B+K]
            adv_loss = F.log_softmax(dot_product, dim=-1) * mask
            adv_loss = adv_loss.sum(dim=1).mean()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            with amp.scale_loss(adv_loss, self.optimizer_adv) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            clip_grad_norm_(amp.master_params(self.optimizer_adv), self.args['grad_clip'])
            self.optimizer.step()
            self.optimizer_adv.step()
            self.scheduler.step()
            self.scheduler_adv.step()

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
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map

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
