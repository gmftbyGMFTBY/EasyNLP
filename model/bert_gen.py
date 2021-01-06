from .header import *
from .base import *

class BERTGen(nn.Module):

    def __init__(self, vocab_size, model='bert-base-chinese', unk_id=0, sep_id=102, cls_id=101):
        super(BERTGen, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.lm_head = nn.Linear(768, vocab_size)
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, S, 768]
        logits = self.lm_head(output)    # [B, S, V]
        return logits
    
    @torch.no_grad()
    def predict(self, inpt, token_type_ids, attn_mask, max_len=50):
        '''
        Due to the BertModel in huggingface doesn"t support past,
        the speed is slow
        inpt: [1, S]; token_type_ids: [1, S], attn_mask: [1, S, S]
        batch size is 1
        '''
        generated = []
        for _ in range(max_len):
            output = self.model(
                inpt, 
                attention_mask=attn_mask, 
                token_type_ids=token_type_ids,
            )[0]    # [B, S, 768]
            logits = self.lm_head(output)    # [1, S, V]
            next_token_logits = logits[0, -1, :]    # [V]
            next_token_logits[self.unk_id] = -np.inf
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            if next_token == self.sep_id:
                break
            # update token_type_ids and attn_mask and inpt
            token_type_ids = torch.cat([token_type_ids, torch.LongTensor([[1]]).cuda()], dim=-1)
            attn_mask = torch.cat([
                attn_mask, 
                torch.LongTensor([0] * attn_mask.shape[1]).view(1, attn_mask.shape[1], 1).cuda()], 
                dim=2)
            attn_mask = torch.cat([attn_mask, torch.LongTensor([1] * attn_mask.shape[2]).view(1, 1, -1).cuda()], dim=1)
            inpt = torch.cat([inpt, next_token.view(1, -1)], dim=1)
        else:
            generated.append(self.sep_id)
        generated = [self.cls_id] + generated
        return generated
    
class BERTGenAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', lang='zh', local_rank=0):
        super(BERTGenAgent, self).__init__()
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
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.args['vocab_size'] = len(self.vocab)
        self.model = BERTGen(self.args['vocab_size'], model=self.args['model'], unk_id=self.unk, sep_id=self.sep)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='sum')
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
        total_loss, batch_num, total_acc, s = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            ids, tids, mask, label = batch
            self.optimizer.zero_grad()
            logits = self.model(ids, tids, mask)    # [B, S, V]
            logits = logits[:, :-1, :]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), 
                label.view(-1),
            )
            
            _, preds = logits.max(dim=-1)
            not_ignore = label.ne(self.pad)
            num_targets = not_ignore.long().sum().item()
            correct = (label == preds) & not_ignore
            correct = correct.float().sum()
            # loss and token accuracy
            accuracy = correct.item() / num_targets
            total_acc += correct.item()
            s += num_targets
            loss = loss / num_targets
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', total_acc/s, idx)
            lr = self.optimizer.param_groups[0]['lr']
            recoder.add_scalar(f'train-epoch-{idx_}/lr', lr, idx)
            
            pbar.set_description(f'[!] lr: {round(lr, 8)}; train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(accuracy, 4)}|{round(total_acc/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, recoder=None):
        self.model.eval()
        def filter_tgt(x):
            if '[SEP]' in x:
                x = x[:x.index('[SEP]')] + '[SEP]'
            return x
        def filter(x):
            return x.replace('[PAD]', '')
        pbar = tqdm(test_iter)
        with open(recoder, 'w') as f:
            for batch in pbar:
                ids, tids, mask, rids = batch
                max_size = max(len(rids[0]), 20)
                generated = self.model.predict(
                    ids, tids, mask, max_len=max_size
                )
                text = self.vocab.convert_ids_to_tokens(generated)
                text = filter_tgt(''.join(text))

                ctx = self.vocab.convert_ids_to_tokens(ids[0])
                ctx = filter(''.join(ctx))

                ref = self.vocab.convert_ids_to_tokens(rids[0])
                ref = filter(''.join(ref))

                f.write(f'CTX: {ctx}\n')
                f.write(f'REF: {ref}\n')
                f.write(f'TGT: {text}\n\n')
        print(f'[!] translate test dataset over, write into {path}')