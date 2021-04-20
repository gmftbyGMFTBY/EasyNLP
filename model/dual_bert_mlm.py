from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', vocab_size=0):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        if model in ['bert-base-uncased']:
            # english corpus has three special tokens: __number__, __url__, __path__
            self.model.resize_token_embeddings(self.model.config.vocab_size + 3)
        self.lm_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, vocab_size)
        )

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        lm_logits = self.lm_head(embd)
        return embd[:, 0, :], lm_logits     # [B, E], [B, S, V]
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    

class BERTDualMLMEncoder(nn.Module):
    
    def __init__(self, model='bert-base-chinese', pad=0, vocab_size=0):
        super(BERTDualMLMEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model, vocab_size=vocab_size)
        self.can_encoder = BertEmbedding(model=model, vocab_size=vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad)
        self.vocab_size = vocab_size

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep, cid_logits = self.ctx_encoder(cid, cid_mask)
        rid_rep, rid_logits = self.can_encoder(rid, rid_mask)
        return cid_rep, cid_logits, rid_rep, rid_logits

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, _ = self.ctx_encoder(cid.unsqueeze(0), None)
        rid_rep, _ = self.can_encoder(rid, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask, cid_label, rid_label):
        batch_size = cid.shape[0]
        cid_rep, cid_logits, rid_rep, rid_logits = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        
        mask = torch.eye(batch_size).cuda()    # [B, B]
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss_1 = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_1.sum(dim=1)).mean()
        
        loss_2 = F.log_softmax(dot_product, dim=0) * mask
        loss += (-loss_2.sum(dim=0)).mean()
        # mlm loss
        masked_lm_loss_ctx = self.criterion(
            cid_logits.view(-1, self.vocab_size), cid_label.view(-1)
        )
        masked_lm_loss_res = self.criterion(
            rid_logits.view(-1, self.vocab_size), rid_label.view(-1)
        )
        mlm_loss = masked_lm_loss_ctx + masked_lm_loss_res
        # add all the losses
        loss += mlm_loss
        return loss, mlm_loss, acc
    
    
class BERTDualMLMEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualMLMEncoderAgent, self).__init__()
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
            'dataset': dataset_name,
            'pretrained_model_path': pretrained_model_path,
            'amp_level': 'O2',
            'test_interval': 0.05,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0

        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.args['vocab_size'] = len(self.vocab)
        self.args['pad'] = self.vocab.convert_tokens_to_ids('[PAD]')
        self.model = BERTDualMLMEncoder(
            model=self.args['model'],
            vocab_size=self.args['vocab_size'],
            pad=self.args['pad'],
        )
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
        total_loss, total_mlm_loss, total_acc, batch_num = 0, 0, 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask, cid_label, rid_label = batch
            loss, mlm_loss, acc = self.model(cid, rid, cid_mask, rid_mask, cid_label, rid_label)
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                # test in the training loop
                index = self.test_step_counter
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
            recoder.add_scalar(f'train-epoch-{idx_}/MLMLoss', total_mlm_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunMLMLoss', mlm_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(total_mlm_loss/batch_num, 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/MLMLoss', total_mlm_loss/batch_num, idx_)
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
