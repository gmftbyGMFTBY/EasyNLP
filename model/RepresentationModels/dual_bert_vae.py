from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)

    def forward(self, ids, attn_mask):
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        return embd[:, 0, :]
    
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
    

class BERTDualEncoderVAE(nn.Module):
    
    def __init__(self, model='bert-base-chinese', times=10, p=0.1):
        super(BERTDualEncoderVAE, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        self.h_to_mu = nn.Linear(768, 768)
        self.h_to_logvar = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )

        self.inference_times = times
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def reparametrize(self, rep):
        z_mu = self.h_to_mu(rep)
        std = self.h_to_logvar(rep)
        eps = torch.FloatTensor(std.size()).normal_().half().cuda()
        z = eps.mul(std).add_(z_mu)
        return z
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        scores = []
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        for _ in range(self.inference_times):
            cid_rep_ = self.reparametrize(cid_rep)
            cid_rep_ = cid_rep_.squeeze(0)    # [768]
            # cid_rep/rid_rep: [768], [B, 768]
            dot_product = torch.matmul(cid_rep_, rid_rep.t())  # [B]
            scores.append(dot_product)
        scores = torch.stack(scores).max(dim=0)[0]
        return scores
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep = self.reparametrize(cid_rep)

        # preserved loss: cid_rep/cid_rep_recons: [B, 768]
        cid_rep_recons = self.decoder(cid_rep)
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean()

        # similarity loss
        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, preserved_loss, acc
    
    
class BERTDualEncoderVAEAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualEncoderVAEAgent, self).__init__()
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
            'times': 5,
            'dropout': 0.1,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualEncoderVAE(model=self.args['model'], times=self.args['times'], p=self.args['dropout'])
        if pretrained_model_path:
            self.load_bert_model(pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        
        )
        if run_mode == 'train':
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
        self.show_parameters(self.args)
        
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num, total_sim_loss, total_pre_loss = 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            sim_loss, preserved_loss, acc = self.model(cid, rid, cid_mask, rid_mask)
            tloss = sim_loss + preserved_loss
            with amp.scale_loss(tloss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += tloss.item()
            total_sim_loss += sim_loss.item()
            total_pre_loss += preserved_loss.item()
            total_acc += acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', tloss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/SimLoss', total_sim_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunSimLoss', sim_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/PreLoss', total_pre_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPreLoss', preserved_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] loss: {round(tloss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss/batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, recoder=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            cid, rids, rids_mask, label = batch
            batch_size = len(rids)
            assert batch_size == 10, f'[!] {batch_size} isnot equal to 10'
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
            total_examples += math.ceil(label.size()[0] / 10)
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        
        for i in range(len(k_list)):
            print(f"R10@{k_list[i]}: {round(((total_correct[i] / total_examples) * 100), 2)}")
        print(f"MRR: {round(avg_mrr, 4)}")
        print(f"P@1: {round(avg_prec_at_one, 4)}")
        print(f"MAP: {round(avg_map, 4)}")

    @torch.no_grad()
    def inference(self, test_iter):
        self.model.eval()
        pbar = tqdm(test_iter)
        matrix, corpus = [], []
        for batch in pbar:
            ids, mask, text = batch
            vec = self.model.get_cand(ids, mask)    # [B, H]
            matrix.append(vec)
            corpus.extend(text)
        matrix = torch.stack(matrix)
        return matrix, corpus
