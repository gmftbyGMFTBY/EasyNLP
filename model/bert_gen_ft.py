from .header import *
from .base import *
from .utils import *

'''BERT Generation and Classification Model to construct the retrieval-based dialog systems (UniLM Mask)'''

class BERTGenFT(nn.Module):

    def __init__(self, vocab_size, model='bert-base-chinese', unk_id=0, sep_id=102, cls_id=101):
        super(BERTGenFT, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.lm_head = nn.Linear(768, vocab_size)
        self.cls_head = nn.Linear(768, 2)
        
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, S, 768]
        lm_logits = self.lm_head(output)    # [B, S, V]
        # collect
        embedding = []
        for idx, item in enumerate(token_type_ids):
            index = item.nonzero().squeeze().tolist()
            # NOTE:
            x, y = index[0] - 1, index[-1]
            v = output[idx][x:y, :]     # [S_, 768]
            embedding.append(v.mean(dim=0))     # [768]
        embedding = torch.stack(embedding)     # [B, 768]
        cls_logits = self.cls_head(embedding)    # [B, 2]
        return lm_logits, cls_logits
    
class BERTGenFTAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', lang='zh', local_rank=0, dataset_name='ecommerce'):
        super(BERTGenFTAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 3e-5,
            'grad_clip': 5.0,
            'multi_gpu': self.gpu_ids,
            'max_len': 256,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'lang': lang,
            'total_step': total_step,
            'warmup_step': warmup_step,
            'dataset': dataset_name,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.args['vocab_size'] = len(self.vocab)
        self.model = BERTGenFT(self.args['vocab_size'], model=self.args['model'], unk_id=self.unk, sep_id=self.sep)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='sum')
        self.cls_criterion = nn.CrossEntropyLoss()
        if run_mode == 'train':
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
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_lm_loss, total_cls_loss, total_loss, batch_num = 0, 0, 0, 0
        items, s, total_token_acc, total_acc = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            ids, tids, mask, cls_label, lm_label = batch
            self.optimizer.zero_grad()
            lm_logits, cls_logits = self.model(ids, tids, mask)    # [B, S, V], [B, 2]
            lm_logits = lm_logits[:, :-1, :]
            lm_loss = self.criterion(
                lm_logits.reshape(-1, lm_logits.size(-1)), 
                lm_label.view(-1),
            )
            cls_loss = self.cls_criterion(
                cls_logits,
                cls_label.view(-1)
            )
            
            # Token Acc (ignore the sample that cls is 0)
            _, preds = lm_logits.max(dim=-1)
            not_ignore = lm_label.ne(self.pad)
            num_targets = not_ignore.long().sum().item()
            correct = (lm_label == preds) & not_ignore
            correct = correct.float().sum()
            # loss and token accuracy
            if num_targets == 0:
                # all the sample in the batch are negative samples
                token_acc = 1
            else:
                token_acc = correct.item() / num_targets
            total_token_acc += correct.item()
            s += num_targets
            lm_loss = lm_loss / num_targets
            
            # CLS acc (total_acc / items)
            now_correct = torch.max(F.softmax(cls_logits, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == cls_label).item()
            total_acc += now_correct
            items += len(cls_label)
            
            # Total loss
            total_lm_loss += lm_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += lm_loss.item() + cls_loss.item()
            batch_num += 1
            
            with amp.scale_loss(lm_loss + cls_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            
            recoder.add_scalar(f'train-epoch-{idx_}/LMLoss', total_lm_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLMLoss', lm_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', token_acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_token_acc/s, idx)
            
            recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/items, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(cls_label), idx)
            
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', lm_loss.item() + cls_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            
            lr = self.optimizer.param_groups[0]['lr']
            recoder.add_scalar(f'train-epoch-{idx_}/lr', lr, idx)
            
            pbar.set_description(f'[!] loss: {round(lm_loss.item() + cls_loss.item(), 4)}|{round(total_loss/batch_num, 4)}; token acc: {round(token_acc, 4)}|{round(total_token_acc/s, 4)}; acc: {round(now_correct/len(cls_label), 4)}|{round(total_acc/items, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/LMLoss', total_lm_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/TokenAcc', total_token_acc/s, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/items, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, recoder=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):
            ids, tids, mask, label = batch
            batch_size = len(ids)
            assert batch_size % 10 == 0, f'[!] {batch_size} cannot mode 10'
            _, scores = self.model(ids, tids, mask)
            scores = scores[:, 1].cpu().tolist()    # [B]
            
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