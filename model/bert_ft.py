from .header import *
from .base import *

'''Cross-Attention BertRetrieval'''

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model, num_labels=2
        )

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        logits = output[0]    # [batch, 2]
        return logits
    
class BERTFTAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', lang='zh', local_rank=0):
        super(BERTFTAgent, self).__init__()
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
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss()
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
            )
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            ids, tids, mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(ids, tids, mask)    # [B, 2]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, recoder=None):
        self.model.eval()
        r1, r2, r5, counter, mrr = 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        with open(recoder, 'w') as f:
            for idx, batch in enumerate(pbar):
                ids, tids, mask, label = batch
                batch_size = len(ids)
                assert batch_size % 10 == 0, f'[!] {batch_size} cannot mode 10'
                scores = self.model(ids, tids, mask).cpu()    # [B, 2]
                scores = scores[:, 1]     # [B]
                for i in range(0, len(label), 10):
                    scores_, label_ = scores[i:i+10], label[i:i+10]
                    ids_ = ids[i:i+10]
                    label_true_ = set(label_.nonzero().squeeze(-1).tolist())
                    r1 += min(1, len(set(torch.topk(scores_, 1, dim=-1)[1].tolist()) & label_true_))
                    r2 += min(1, len(set(torch.topk(scores_, 2, dim=-1)[1].tolist()) & label_true_))
                    r5 += min(1, len(set(torch.topk(scores_, 5, dim=-1)[1].tolist()) & label_true_))
                    # mrr
                    scores_ = scores_.numpy()
                    y_true = np.zeros(len(scores_))
                    for item in label_true_:
                        y_true[item] = 1
                    mrr.append(label_ranking_average_precision_score([y_true], [scores_]))
                    counter += 1

                    # write the response selection results
                    for s, text in zip(scores_, ids_):
                        text = self.vocab.decode(text).replace('[PAD]', '')
                        f.write(f'[{s}]\t{text}\n')
                    f.write('\n')
            
        r1, r2, r5, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; mrr: {mrr}')
        print(f'[!] results are saved into {recoder}')