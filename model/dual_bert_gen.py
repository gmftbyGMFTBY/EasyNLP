from .header import *
from .base import *
from .utils import *


class BertGen(nn.Module):

    def __init__(self, vocab_size, model='bert-base-chinese', unk_id=0, sep_id=102):
        super(BertGen, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.lm_head = nn.Linear(768, vocab_size)
        self.fusion_head = nn.Linear(768*2, 768)
        self.unk_id, self.sep_id = unk_id, sep_id

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, S, 768]
        lm_logits = self.lm_head(output)    # [B, S, V]
        embedding = []
        for idx, item in enumerate(token_type_ids):
            index = item.nonzero().squeeze(-1).tolist()
            x, y = index[0], index[-1] + 1
            v = output[idx][x:y, :]
            embedding.append(v.mean(dim=0))    # [768]
        embedding = torch.stack(embedding)    # [B, 768]
        cls_logits = self.fusion_head(
            torch.cat([embedding, output[:, 0, :]], dim=1)
        )    # [B, 768]
        return lm_logits, cls_logits

    @torch.no_grad()
    def predict(self, inpt, token_type_ids, attn_mask, max_len=64):
        embedding = []
        generation_rest = []
        for _ in range(max_len):
            output = self.model(inpt, attention_mask=attn_mask, token_type_ids=token_type_ids)[0]    # [B, S, 768]
            logits = self.lm_head(output)
            next_token_logits = logits[0, -1, :]
            embedding.append(next_token_logits)    # [768]
            next_token_logits[self.unk_id] = -np.inf
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1),
                num_samples=1,
            )
            generation_rest.append(next_token)
            if next_token == self.sep_id:
                break
            token_type_ids = torch.cat([
                token_type_ids, 
                torch.LongTensor([[1]]).cuda()], dim=-1
            )
            attn_mask = torch.cat([
                attn_mask,
                torch.LongTensor([0] * attn_mask.shape[1]).view(1, attn_mask.shape[1], 1).cuda()], dim=2
            )
            attn_mask = torch.cat([attn_mask, torch.LongTensot([1] * attn_mask.shape[2]).view(1, 1, -1).cuda()], dim=1)
            inpt = torch.cat([inpt, next_token.view(1, -1)], dim=1)
        embedding = torch.stack(embedding).mean(dim=0)    # [S, 768] -> [768]
        cls_logits = torch.cat([embedding, output[0, 0, :]])    # [768]
        return cls_logits, generation
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return embd[:, 0, :]
    
    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    

class BERTDualGenEncoder(nn.Module):
    
    def __init__(self, vocab_size, model='bert-base-chinese', unk_id=0, sep_id=102, pad_id=0, res_max_len=64):
        super(BERTDualGenEncoder, self).__init__()
        self.ctx_encoder = BertGen(vocab_size, model=model, unk_id=unk_id, sep_id=sep_id)
        self.can_encoder = BertEmbedding(model=model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.pad = pad_id
        self.res_max_len = res_max_len

    def _encode(self, cid, rid, cid_mask, rid_mask, tid):
        lm_logits, cls_logits = self.ctx_encoder(cid, tid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return lm_logits, cls_logits, rid_rep
    
    def _encode_(self, cid, rid, cid_mask, rid_mask, tid):
        cls_logits, generation = self.ctx_encoder.predict(cid, tid, cid_mask, maxlen=self.res_max_len)
        rid_rep = self.can_encoder(rid, rid_mask)
        return generation, cls_logits, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid).cuda()
        generation, cid_rep, rid_rep = self._encode_(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product, generation
        
    def forward(self, cid, rid, cid_mask, rid_mask, token_type_ids, label):
        batch_size = cid.shape[0]
        lm_logits, cls_logits, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, token_type_ids)

        # ========== GEN LOSS ========== #
        lm_logits = lm_logits[:, :-1, :]
        lm_loss = self.criterion(
            lm_logits.reshape(-1, lm_logits.size(-1)) ,
            label.view(-1),
        )
        _, preds = lm_logits.max(dim=-1)
        not_ignore = label.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (label == preds) & not_ignore
        correct = correct.float().sum()
        lm_acc = correct.item() / num_targets
        lm_loss = lm_loss / num_targets

        # ========== CLS LOSS ========== #
        dot_product = torch.matmul(cls_logits, rid_rep.t())  # [B, B]
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        cls_acc = acc_num / batch_size
        # calculate the loss
        cls_loss = F.log_softmax(dot_product, dim=-1) * mask
        cls_loss = (-cls_loss.sum(dim=1)).mean()
        return lm_loss, lm_acc, cls_loss, cls_acc
    
    
class BERTDualGenEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualGenEncoderAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.vocab = BertTokenizer.from_pretrained(pretrained_model)
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
            'vocab_size': len(self.vocab),
            'test_interval': 0.05,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0

        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.model = BERTDualGenEncoder(self.args['vocab_size'], model=self.args['model'], unk_id=self.unk, sep_id=self.sep, pad_id=self.pad)
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
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_lm_loss, total_cls_loss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, tid, rid, cid_mask, rid_mask, label = batch
            lm_loss, lm_acc, cls_loss, cls_acc = self.model(cid, rid, cid_mask, rid_mask, tid, label)
            loss = lm_loss + cls_loss
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
    
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_cls_loss += cls_loss.item()
            total_acc += cls_acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                index = self.test_step_counter
                (r10_1, r10_2, r10_5), avg_mrr, avg_P1, avg_map = self.test_model()
                self.model.train()
                recoder.add_scalar(f'train-test/R10@1', r10_1, index)
                recoder.add_scalar(f'train-test/R10@2', r10_2, index)
                recoder.add_scalar(f'train-test/R10@5', r10_5, index)
                recoder.add_scalar(f'train-test/MRR', avg_mrr, index)
                recoder.add_scalar(f'train-test/P@1', avg_p1, index)
                recoder.add_scalar(f'train-test/MAP', avg_map, index)
                # random sample 10 generated samples
                generated_rest = random.sample(self.collections, 10)
                for i, (ctx, res) in enumerate(generated_rest):
                    recoder.add_scalar(
                        f'train-test/generation-{i}', f'{ctx} [SEP] {res}', index
                    )
                self.test_step_counter += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/LMLoss', total_lm_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLMLoss', lm_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', cls_acc, idx)
            
            pbar.set_description(f'[!] lm_loss: {round(lm_loss.item(), 4)}|{round(total_lm_loss/batch_num, 4)}; loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(cls_acc, 4)}|{round(total_acc/batch_num, 4)}')

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/LMLoss', total_lm_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        pbar = tqdm(self.test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]

        self.collections = []
        for idx, batch in enumerate(pbar):                
            cid, rids, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} is not equal to 10'
            scores, generation = self.model.module.predict(cid, rids, rids_mask).cpu().tolist()    # [B]
            # collect the context and the generated response
            ctx = self.vocab.decode(cid)
            res = self.vocab.decode(generation)
            self.collections.append((ctx, res))

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
