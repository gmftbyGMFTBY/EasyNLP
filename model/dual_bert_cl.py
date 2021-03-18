from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
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

    def __init__(self, model='bert-base-chinese', T=0.07, K=4096):
        super(BERTDualCLEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # queue
        self.K = K
        self.T = T
        self.register_buffer('queue', torch.randn(self.K, 768))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        '''collect samples from all the devices'''
        tensors_gather = [torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = self.concat_all_gather(keys)
        batch_size = len(keys)
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[ptr:ptr+batch_size, :] = keys
        else:
            s_before = self.K - ptr
            s_after = batch_size - (self.K - ptr)
            self.queue[ptr:, :] = keys[:s_before, :]
            self.queue[:s_after, :] = keys[-s_after:, :]
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        # cid_rep/rid_rep: [768], [B, 768]
        cid_rep = cid_rep.squeeze(0)    # [768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        acc, loss = 0, 0

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        neg_rep = self.queue.clone().detach()    # [K, 768]
        
        dot_product_1 = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        dot_product_2 = torch.matmul(cid_rep, neg_rep.t())    # [B, K]
        dot_product = torch.cat([dot_product_1, dot_product_2], dim=1)    # [B, B+K]
        dot_product /= self.T
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        zeros = torch.zeros_like(dot_product_2).cuda().half()
        mask = torch.cat([mask, zeros], dim=1)
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # calculate the acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # dequeue and enqueue, buffer tensor will not be updated by optimizer
        self._dequeue_and_enqueue(rid_rep)
        return loss, acc, int(self.queue_ptr)
    
    
class BERTDualCLEncoderAgent(RetrievalBaseAgent):

    '''response encoder update slower'''
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None, head=10):
        super(BERTDualCLEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'res_lr': 2e-5,
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
            'T': 0.07,
            'K': 65536,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualCLEncoder(model=self.args['model'], T=self.args['T'], K=self.args['K'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer_ctx = transformers.AdamW(
            self.model.ctx_encoder.parameters(), 
            lr=self.args['lr'],
        )
        self.optimizer_res = transformers.AdamW(
            self.model.can_encoder.parameters(), 
            lr=self.args['res_lr'],
        )
        if run_mode == 'train':
            self.model, [self.optimizer_ctx, self.optimizer_res] = amp.initialize(
                self.model, 
                [self.optimizer_ctx, self.optimizer_res],
                opt_level=self.args['amp_level'],
            )
            self.scheduler_ctx = transformers.get_linear_schedule_with_warmup(
                self.optimizer_ctx, 
                num_warmup_steps=warmup_step, 
                num_training_steps=total_step,
            )
            self.scheduler_res = transformers.get_linear_schedule_with_warmup(
                self.optimizer_res, 
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
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer_ctx.zero_grad()
            self.optimizer_res.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            loss, acc, ptr = self.model(cid, rid, cid_mask, rid_mask)
            with amp.scale_loss(loss, [self.optimizer_ctx, self.optimizer_res]) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer_ctx), self.args['grad_clip'])
            clip_grad_norm_(amp.master_params(self.optimizer_res), self.args['grad_clip'])
            self.optimizer_ctx.step()
            self.optimizer_res.step()
            self.scheduler_ctx.step()
            self.scheduler_res.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            
            pbar.set_description(f'[!] ptr: {ptr}|{self.args["K"]}; loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

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
            total_examples += math.ceil(label.size()[0] / 10)
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        
        # for i in range(len(k_list)):
        #     print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        # print(f"MRR: {round(avg_mrr, 4)}")
        # print(f"P@1: {round(avg_prec_at_one, 4)}")
        # print(f"MAP: {round(avg_map, 4)}")
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
