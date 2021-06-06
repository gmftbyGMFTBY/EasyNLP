from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    def forward(self, ids, attn_mask):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return embd

    def load_bert_model(self, state_dict):
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if k.startswith('_bert_model.cls.'):
        #         continue
        #    name = k.replace('_bert_model.bert.', '')
        #     new_state_dict[name] = v
        # self.model.load_state_dict(new_state_dict)

        # load the post train checkpoint from BERT-FP (NAACL 2021)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)


class PolyEncoder(nn.Module):
    
    def __init__(self, m=16, model='bert-base-chinese'):
        super(PolyEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        # m global queries
        self.ctx_poly_embd = nn.Embedding(m, 768)
        self.res_poly_embd = nn.Embedding(m, 768)
        self.m = m

    def poly(self, id_rep, query_id, id_mask):
        poly_query = self.ctx_poly_embd(query_id)    # [M, E]
        # cid_rep: [B, S, E]; [M, E]
        weights = torch.matmul(id_rep, poly_query.t()).permute(0, 2, 1)    # [B, M, S]
        weights /= np.sqrt(768)
        id_mask_ = torch.where(id_mask != 0, torch.zeros_like(id_mask), torch.ones_like(id_mask))
        id_mask_ = id_mask_ * -1e3
        id_mask_ = id_mask_.unsqueeze(1).repeat(1, self.m, 1)    # [B, M, S]
        weights += id_mask_
        weights = F.softmax(weights, dim=-1)

        id_rep = torch.bmm(
            weights,     # [B, M, S]
            id_rep,     # [B, S, E]
        )    # [B, M, E]
        return id_rep
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        poly_query_id = torch.arange(self.m, dtype=torch.long).cuda()

        rid_rep = self.can_encoder(rid, rid_mask)
        rid_rep = rid_rep[:, 0, :]    # [B, E]
        # cid_rep: [B, M, E]; rid_rep: [B, E]
        return cid_rep, rid_rep
        
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_mask = torch.ones(1, len(cid)).cuda()
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, cid_mask, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [M, E]
        # cid_rep/rid_rep: [M, E], [B, E]
        
        # POLY ENCODER ATTENTION
        # [M, E] X [E, S] -> [M, S] -> [S, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1)
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        # [S, M] X [M, E] -> [S, E]
        cid_rep = torch.matmul(weights, cid_rep)
        dot_product = (cid_rep * rid_rep).sum(-1)    # [S]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, M, E]; [B, E]
        
        # POLY ENCODER ATTENTION
        # [B, M, E] X [E, B] -> [B, M, B]-> [B, B, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1)    # [B, M, B] -> [B, B, M]
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)    # [B, B, M] X [B, M, E] -> [B, B, E]
        # [B, B, E] x [B, B, E] -> [B, B]
        dot_product = (cid_rep * rid_rep.unsqueeze(0).expand(batch_size, -1, -1)).sum(-1)
        mask = torch.eye(batch_size).cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc

    
class BERTPolyEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTPolyEncoderAgent, self).__init__()
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
            'm': 16,
            'test_interval': 0.05
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = PolyEncoder(m=self.args['m'], model=self.args['model'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        if run_mode in ['train', 'train-post', 'train-dual-post']:
            self.scaler = GradScaler()
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
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            with autocast():
                loss, acc = self.model(cid, rid, cid_mask, rid_mask)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} isnot equal to 10'
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
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map
        
        
