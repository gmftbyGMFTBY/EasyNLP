from .header import *
from .base import *
from .utils import *


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertForPreTraining.from_pretrained(model)
        self.model.cls.seq_relationship = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, ids, attn_mask):
        output = self.model(ids, attention_mask=attn_mask)
        return output.prediction_logits, output.seq_relationship_logits


class BERTDualPretrainEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, model='bert-base-chinese', s=0.1, vocab_size=21128):
        super(BERTDualPretrainEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_pred_logits, cid_embd_logits = self.ctx_encoder(cid, cid_mask)
        rid_pred_logits, rid_embd_logits = self.can_encoder(rid, rid_mask)
        return cid_pred_logits, cid_embd_logits, rid_pred_logits, rid_embd_logits

    def forward(self, cid, rid, cid_mask, rid_mask, cid_label, rid_label):
        batch_size = cid.shape[0]
        cid_pred_logits, cid_embd_logits, rid_pred_logits, rid_embd_logits = self._encode(cid, rid, cid_mask, rid_mask)

        # MLM Loss
        cid_mlm_loss = self.mlm_loss(
            cid_pred_logits.view(-1, self.vocab_size),
            cid_label.view(-1)
        )
        rid_mlm_loss = self.mlm_loss(
            rid_pred_logits.view(-1, self.vocab_size),
            rid_label.view(-1)
        )

        # Constrastive Loss
        # cid_embd_logits/rid_embd_logits: [B, E]
        gold = torch.arange(batch_size).cuda()
        dot_product = torch.matmul(cid_embd_logits, rid_embd_logits.t())     # [B, B]
        dot_product /= np.sqrt(768)     # scale dot product
        cl_loss = self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # total loss
        loss = cl_loss + cid_mlm_loss + rid_mlm_loss
        return loss, acc
    
    
class BERTDualPretrainEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, warmup_step, run_mode='train', local_rank=0, dataset_name='ecommerce', pretrained_model='bert-base-chinese', pretrained_model_path=None):
        super(BERTDualPretrainEncoderAgent, self).__init__()
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
            'dropout': 0.1,
            'amp_level': 'O2',
            'smoothing': 0.1,
            'run_mode': run_mode,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['model'])
        self.model = BERTDualPretrainEncoder(
            model=self.args['model'], 
            s=self.args['smoothing'],
            vocab_size=len(self.vocab),
        )
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
        elif run_mode in ['inference', 'inference_qa']:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask, cid_label, rid_label = batch
            with autocast():
                loss, acc = self.model(cid, rid, cid_mask, rid_mask, cid_label, rid_label)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
