from .header import *
from .base import *
from .utils import *


'''fine-grained interaction among context and response'''


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', p=0.2):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        if model in ['bert-base-uncased']:
            # english corpus has three special tokens: __number__, __url__, __path__
            self.model.resize_token_embeddings(self.model.config.vocab_size + 3)

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return embd[:, 0, :], embd
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    

class BERTDualFGEncoder(nn.Module):
    
    def __init__(self, model='bert-base-chinese', scale_loss=1/32, p=0.2, lambd=3.9e-3):
        super(BERTDualFGEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model, p=p)
        self.can_encoder = BertEmbedding(model=model, p=p)
        self.lambd = lambd
        self.scale_loss = scale_loss
        self.bn = nn.BatchNorm1d(768, affine=False)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep, cid_rep_h = self.ctx_encoder(cid, cid_mask)
        rid_rep, rid_rep_h = self.can_encoder(rid, rid_mask)
        return (cid_rep, rid_rep), (cid_rep_h, rid_rep_h)

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    def get_dot_product(self, cid_rep, cid_rep_h, rid_rep, rid_rep_h, cid_mask, rid_mask, ctx_batch_size, res_batch_size):
        # 1. from response to context
        weight_r2c = torch.bmm(
            rid_rep.unsqueeze(0).repeat(ctx_batch_size, 1, 1).detach(),    # [B_c, B_r, E]
            cid_rep_h.permute(0, 2, 1),     # [B_c, E, S_c]
        )    # [B_c, B_r, S_c]
        # scale is very important
        weight_r2c /= np.sqrt(768)
        weight_c2r = torch.bmm(
            cid_rep.unsqueeze(0).repeat(res_batch_size, 1, 1).detach(),    # [B_r, B_c, E]
            rid_rep_h.permute(0, 2, 1),     # [B_r, E, S_r]
        )    # [B_r, B_c, S_r]
        weight_c2r /= np.sqrt(768)
        # 2. mask and softmax
        cid_mask_ = torch.where(
            cid_mask != 0, 
            torch.zeros_like(cid_mask), 
            torch.ones_like(cid_mask)
        )
        cid_mask_ = cid_mask_ * -1e3    # [B_c, S_c]
        cid_mask_ = cid_mask_.unsqueeze(1).repeat(1, res_batch_size, 1)    # [B_c, B_r, S_c]
        weight_r2c += cid_mask_
        weight_r2c = F.softmax(weight_r2c, dim=-1)

        rid_mask_ = torch.where(
            rid_mask != 0, 
            torch.zeros_like(rid_mask), 
            torch.ones_like(rid_mask)
        )
        rid_mask_ = rid_mask_ * -1e3    # [B_r, S_r]
        rid_mask_ = rid_mask_.unsqueeze(1).repeat(1, ctx_batch_size, 1)    # [B_r, B_c, S_r]
        weight_c2r += rid_mask_
        weight_c2r = F.softmax(weight_c2r, dim=-1)
        # 3. attention
        cid_rep_ = torch.bmm(
            weight_r2c,    # [B_c, B_r, S_c]
            cid_rep_h,    # [B_c, S_c, E]
        )    # [B_c, B_r, E]
        rid_rep_ = torch.bmm(
            weight_c2r,    # [B_r, B_c, S_r]
            rid_rep_h,    # [B_r, S_r, E]
        )    # [B_r, B_c, E]
        # 4. residual
        cid_rep_ += cid_rep.unsqueeze(1).repeat(1, res_batch_size, 1)
        rid_rep_ += rid_rep.unsqueeze(1).repeat(1, ctx_batch_size, 1)
        # 5. dot product
        rid_rep_ = rid_rep_.permute(1, 0, 2)    # [B_c, B_r, E]
        dot_product = torch.einsum('ijk,ijk->ij', cid_rep_, rid_rep_)
        return dot_product    # [B_c, B_r]
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        ctx_batch_size, res_batch_size = 1, batch_size
        (cid_rep, rid_rep), (cid_rep_h, rid_rep_h) = self._encode(cid.unsqueeze(0), rid, None, rid_mask)     # [B_c, E]; [B_c, S_c, E]; [B_r, E]; [B_r, S_r, E]
        # cid_mask: [B_c, S], full one
        cid_mask = torch.ones(1, len(cid)).cuda()
        dot_product = self.get_dot_product(cid_rep, cid_rep_h, rid_rep, rid_rep_h, cid_mask, rid_mask, ctx_batch_size, res_batch_size)    # [B_c, B_r] = [1, B_r]
        dot_product = dot_product.squeeze(0)    # [B_r]
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        '''add barlow twins loss'''
        batch_size = cid.shape[0]
        ctx_batch_size, res_batch_size = batch_size, batch_size
        (cid_rep, rid_rep), (cid_rep_h, rid_rep_h) = self._encode(cid, rid, cid_mask, rid_mask)    # [B, S, E]
        dot_product = self.get_dot_product(cid_rep, cid_rep_h, rid_rep, rid_rep_h, cid_mask, rid_mask, ctx_batch_size, res_batch_size)

        mask = torch.eye(batch_size).cuda()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
    
class BERTDualFGEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualFGEncoderAgent, self).__init__()
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
            'dropout': 0.2,
            'amp_level': 'O2',
            'test_interval': 0.05,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0

        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualFGEncoder(model=self.args['model'], p=self.args['dropout'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
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
        elif run_mode in ['inference']:
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
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
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
                # test in the training loop
                index = self.test_step_counter
                index = self.args['test_step'].index(batch_num)
                (r10_1, r10_2, r10_5), avg_mrr, avg_p1, avg_map = self.test_model()
                self.model.train()    # reset the train mode
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
