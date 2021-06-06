from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        if model in ['bert-base-uncased']:
            self.model.resize_token_embeddings(self.model.config.vocab_size + 3)
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask)[0]
        embds = embds[:, 0, :]     # [CLS]
        return embds

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)


class HashBERTModel(nn.Module):
    
    def __init__(self, model, hidden_size, hash_code_size, dropout=0.1):
        super(HashBERTModel, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.hash_code_size = hash_code_size
        
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid)
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep)) 
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t())   # [B]
        # hamming distance [B]: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j)
        # distance = 0.5 * (self.hash_code_size - matrix)
        return matrix
        
    def forward(self, cid, cid_mask, rid, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        batch_size = cid_rep.shape[0]
        
        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, Hash]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B, Hash]
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h = torch.sign(ctx_hash_code).detach()
        can_hash_code_h = torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)    # [B, B]
        label_matrix = self.hash_code_size * torch.eye(batch_size).cuda()
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean()
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())
        # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j)
        hamming_distance = 0.5 * (self.hash_code_size - matrix)
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        return acc, hash_loss, quantization_loss


class HashBertAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(HashBertAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'local_rank': local_rank,
            'dropout': 0.1,
            'hidden_size': 512,
            'hash_code_size': 1024,
            'total_steps': total_step,
            'samples': 10,
            'amp_level': 'O2',
            'q_alpha': 1e-4,
            'q_alpha_max': 1e-1,
            'pretrained_model': pretrained_model,
            'pretrained_model_path': pretrained_model_path,
            'test_interval': 0.05,
            'run_mode': run_mode,
            'dataset': dataset_name,
        }
        self.args['test_step'] = [int(total_step*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        
        self.model = HashBERTModel(
            self.args['pretrained_model'],
            self.args['hidden_size'],
            self.args['hash_code_size'],
            dropout=self.args['dropout'],
        )
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(),
            lr=self.args['lr']
        )
        if run_mode in ['train', 'train-post', 'train-dual-post']:
            self.args['q_alpha_step'] = (self.args['q_alpha_max'] - self.args['q_alpha']) / int(total_step / len(self.gpu_ids))
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
        elif run_mode == 'inferance':
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
        pprint.pprint(self.args)
        
    def load_bert_model(self, path):
        # load the bert encoder
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load context and response bert model from {path}')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_acc, total_loss, total_h_loss, total_q_loss, batch_num = 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            with autocast():
                acc, hash_loss, quantization_loss = self.model(
                    cid, cid_mask,
                    rid, rid_mask,
                )
                quantization_loss = self.args['q_alpha'] * quantization_loss
                loss = hash_loss + quantization_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.args['q_alpha'] += self.args['q_alpha_step']
            total_loss += loss.item()
            total_acc += acc
            total_q_loss += quantization_loss.item()
            total_h_loss += hash_loss.item()
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
            
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationAlpha', self.args['q_alpha'], idx)
            
            pbar.set_description(f'[!] loss: {round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QuantizationLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HashLoss', total_h_loss/batch_num, idx_)
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
            if self.args['run_mode'] in ['train', 'train-post', 'train-dual-post']:
                scores = self.model.module.predict(cid, rids, rids_mask).cpu().tolist()    # [B]
            else:
                scores = self.model.predict(cid, rids, rids_mask).cpu().tolist()    # [B]


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
