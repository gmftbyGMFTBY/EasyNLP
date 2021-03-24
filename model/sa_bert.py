from .header import *
from .base import *
from .utils import *

'''Speraker-aware Bert Cross-encoder'''

class SABERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(SABERTRetrieval, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)    # [EOT]
        self.head = nn.Linear(768, 2)
        self.speaker_embedding = nn.Embedding(2, 768)

    def forward(self, inpt, token_type_ids, speaker_type_ids, attn_mask):
        word_embeddings = self.model.embeddings(inpt)    # [B, S, 768]
        speaker_embedding = self.speaker_embedding(speaker_type_ids)   # [B, S, 768]
        word_embeddings += speaker_embedding
        output = self.model(
            input_ids=None,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=word_embeddings,
        )[0]    # [B, S, E]
        logits = self.head(output[:, 0, :])    # [B, H] -> [B, 2]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_bert_model.cls.'):
                continue
            name = k.replace('_bert_model.bert.', '')
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    
class SABERTFTAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', pretrained_model='bert-base-chinese', local_rank=0, dataset_name='ecommerce', pretrained_model_path=None):
        super(SABERTFTAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 3e-5,
            'grad_clip': 5.0,
            'multi_gpu': self.gpu_ids,
            'max_len': 256,
            'model': pretrained_model,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'total_step': total_step,
            'warmup_step': warmup_step,
            'dataset': dataset_name,
            'pretrained_model_path': pretrained_model_path,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.vocab.add_tokens(['[EOT]'])
        self.model = SABERTRetrieval(self.args['model'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
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
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            ids, tids, sids, mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(ids, tids, sids, mask)    # [B, 2]
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
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):
            ids, tids, sids, mask, label = batch
            batch_size = len(ids)
            assert batch_size % 10 == 0, f'[!] {batch_size} cannot mode 10'
            scores = self.model(ids, tids, sids, mask)[:, 1].tolist()    # [B]
            rank_by_pred, pos_index, stack_scores = \
                    calculate_candidates_ranking(
                        np.array(scores), 
                        np.array(label.cpu().tolist()),
                        10
                    )
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