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
        return rest, embd

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    
class BERTDualCompEncoder(nn.Module):
    
    def __init__(self, nhead, dropout=0.1, model='bert-base-chinese', K=65536, topk=64, cache=256):
        super(BERTDualCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        self.trs_encoder = nn.MultiheadAttention(768, nhead, dropout=dropout)
        self.proj1 = nn.Linear(768*2, 768)
        self.gate = nn.Linear(768*2, 768)

        self.K = K
        self.topk = topk
        self.cache = cache
        self.register_buffer('queue', torch.randn(self.K, 768))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep, _ = self.ctx_encoder(cid, cid_mask)
        _, rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        tensor = tensor.contiguous()
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
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

    def get_dot_product(self, cid_rep, rid_rep, rid_mask, mode='train'):
        '''cid_rep: [B_c, E]; rid_rep: [B_r, S, E]; rid_mask: [B_r, S]'''
        # expand the negative samples with hard samples from the queue
        ctx_batch_size, res_batch_size = len(cid_rep), len(rid_rep)
        if mode == 'train':
            with torch.no_grad():
                key = torch.matmul(cid_rep, self.queue.t())    # [B_c, K]
                key = key.topk(self.topk, dim=1)[1]     # [B_c, topk]
                hard_rep = []
                for index in key:
                    hard_rep.append(self.queue[index, :])
                hard_rep = torch.stack(hard_rep)     # [B_c, K, E]
            # reconstruct the rid_rep
            rid_rep = torch.cat([
                rid_rep.unsqueeze(0).repeat(ctx_batch_size, 1, 1),    # [B_c, B_r, E]
                hard_rep,    # [B_c, K, E] 
                ], dim=1
            )    # [B_c, K+B_r, E]
            topk = self.topk
        else:
            topk = 0
        # context: [B_c, E]; rid_rep_h: [B_r, S, E]
        weights = torch.matmul(rid_rep, cid_rep.t()).permute(0, 2, 1)    # [B_r, B_c, S]
        weights /= np.sqrt(768)
        rid_mask = rid_mask.unsqueeze(1).repeat(1, ctx_batch_size, 1)    # [B_r, B_c, S]
        rid_mask_ = torch.where(rid_mask != 0, torch.zeros_like(rid_mask), torch.ones_like(rid_mask))
        rid_mask_ = rid_mask_ * -1e3
        weights += rid_mask_
        weights = F.softmax(weights, dim=-1)
        rid_rep = torch.bmm(weights, rid_rep).permute(1, 0, 2)    # [B_c, B_r, E]
        rid_rep = torch.tanh(
            self.proj1(torch.cat([
                cid_rep.unsqueeze(1).repeat(1, res_batch_size, 1), 
                rid_rep], dim=-1)
            )
        )    # [B_c, B_r, E]
        rid_rep = rid_rep.permute(1, 0, 2)    # [B_r, B_c, E]
        # forbiden the self-information
        attn_mask = torch.eye(topk+res_batch_size).to(torch.bool).cuda()
        cross_rep = self.trs_encoder(
            rid_rep, rid_rep, rid_rep, attn_mask=attn_mask,
        )[0].permute(1, 0, 2) # [B_c, B_r, E]
        
        gate = torch.sigmoid(
            self.gate(torch.cat([rid_rep.permute(1, 0, 2), cross_rep], dim=-1))
        )    # [B_c, K+B_r, E]
        cross_rep = gate * rid_rep.permute(1, 0, 2) + (1 - gate) * cross_rep    # [B_c,B_r,E]
        # dot_product: [B_c, B_r+K]; cid_rep: [B_c, E]; cross_rep: [B_c, B_r+K, E]
        cid_rep = cid_rep.unsqueeze(1)    # [B_c, 1, E]
        dot_product = torch.bmm(
            cid_rep, 
            cross_rep.permute(0, 2, 1)
        ).squeeze(1)    # [B_c, K+B_r]
        return dot_product
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        # cid_rep: [1, E]; rid_rep: [S, E]
        batch_size = rid.shape[0]
        cid = cid.unsqueeze(0)
        cid_rep, rid_rep = self._encode(cid, rid, None, rid_mask)
        dot_product = self.get_dot_product(cid_rep, rid_rep, rid_mask, mode='test')    # [B_c, B_r]
        return dot_product.squeeze(0)    # [B_r]
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, E]

        dot_product = self.get_dot_product(cid_rep, rid_rep, rid_mask, mode='test')    # [B_c, B_r+K]
        mask = torch.zeros_like(dot_product).cuda()
        mask[torch.arange(batch_size), torch.arange(batch_size)] = 1
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        # dequeue and enqueue
        # self._dequeue_and_enqueue(rid_rep)
        return loss, acc, int(self.queue_ptr)

    
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
            'dropout': 0.,
            'test_interval': 0.05,
            'model': pretrained_model,
            'pretrained_model_path': pretrained_model_path,
            'dataset': dataset_name,
            'K': 65536,
            'topk': 128,
            'cache': 512,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualCompEncoder(
            self.args['nhead'], 
            dropout=self.args['dropout'],
            model=self.args['model'],
            K=self.args['K'],
            topk=self.args['topk'],
            cache=self.args['cache'],
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
            loss, acc, ptr = self.model(cid, rid, cid_mask, rid_mask)
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

            pbar.set_description(f'[!] ptr: {ptr}|{self.args["K"]}; loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
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
