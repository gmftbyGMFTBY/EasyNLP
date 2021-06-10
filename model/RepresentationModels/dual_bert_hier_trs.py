from model.utils import *


class BERTDualHierarchicalTrsEncoder(nn.Module):

    def __init__(self, model='bert-base-chinese', nlayer=3, nhead=6, nhide=512, dropout=0.1):
        super(BERTDualHierarchicalTrsEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            768,
            nhead=nhead,
            dim_feedforward=nhide,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(768)
        self.position_embd = nn.Embedding(512, 768)
        self.speaker_embd = nn.Embedding(2, 768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            nlayer,
            encoder_norm,
        )

    def _encode(self, cids, rid, cids_mask, rid_mask, recover_mapping):
        '''resort'''
        cid_reps = []
        for cid, cid_mask in zip(cids, cids_mask):
            cid_rep = self.ctx_encoder(cid, cid_mask)
            cid_reps.append(cid_rep)
        cid_reps = torch.cat(cid_reps)    # [B, E]
        # recover
        cid_reps = [cid_reps[recover_mapping[idx]] for idx in range(len(cid_reps))]
        cid_rep = torch.stack(cid_reps)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def _encode_(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def reconstruct_tensor(self, cid_rep, cid_turn_length):
        '''resort and generate the order, context length mask'''
        # =========== reconstruct cid ========== #
        cid_rep = torch.split(cid_rep, cid_turn_length)
        # =========== padding =========== #
        max_turn_length = max([len(i) for i in cid_rep])
        cid_reps = []    # [B, S, E]
        cid_mask = []    # [B, S]
        for ctx in cid_rep:
            # mask, [S]
            m = torch.tensor([0] * len(ctx) + [1] * (max_turn_length - len(ctx))).to(torch.bool)
            cid_mask.append(m)
            if len(ctx) < max_turn_length:
                # support apex
                zero_tensor = torch.zeros(1, 768).half().cuda()
                padding = [zero_tensor] * (max_turn_length - len(ctx))
                ctx = torch.cat([ctx] + padding)    # append [S, E]
            cid_reps.append(ctx)
        pos_index = torch.arange(max_turn_length).repeat(len(cid_rep), 1).cuda()    # [B, S]
        cid_reps = torch.stack(cid_reps)
        cid_mask = torch.stack(cid_mask).cuda()
        spk_index = torch.ones(len(cid_rep), max_turn_length).cuda()    # [B, S]
        spk_index[:, ::2] = 0
        spk_index = spk_index.to(torch.long)
        return cid_reps, cid_mask, pos_index, spk_index  # [B, S, E], [B, S], [B, S]
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_turn_length, cid_mask, rid_mask):
        '''batch size is 1'''
        batch_size = rid.shape[0]
        
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # [S, E], [10, E]
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)
        
        pos_embd = self.position_embd(pos_index)
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [1, S, E]

        cid_rep += cid_rep_base
        cid_rep = cid_rep[:, cid_turn_length-1, :]    # [1, E]
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()    # [10] 
        return dot_product

    def forward(self, cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, recover_mapping)
        cid_rep_base, cid_mask, pos_index, spk_index = self.reconstruct_tensor(cid_rep, cid_turn_length)

        # Transformer Encoder
        pos_embd = self.position_embd(pos_index)    # [B, S, E]
        spk_embd = self.speaker_embd(spk_index)
        cid_rep = cid_rep_base + pos_embd + spk_embd

        cid_rep = self.trs_encoder(cid_rep.permute(1, 0, 2), src_key_padding_mask=cid_mask).permute(1, 0, 2)    # [B, S, E]

        cid_rep += cid_rep_base

        last_utterance = []
        for i in range(len(cid_turn_length)):
            c = cid_rep[i]
            p = cid_turn_length[i]
            last_utterance.append(c[p-1, :])
        cid_rep = torch.stack(last_utterance)    # [B_c, E]

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        mask = torch.eye(batch_size).cuda().half()    # [B, B]
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        loss_1 = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_1.sum(dim=1)).mean()
        return loss, acc
        
    
class BERTDualHierarchicalTrsEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, args):
        super(BERTDualHierarchicalTrsEncoderAgent, self).__init__()
        self.args = args
        self.set_test_interval()
        self.vocab = BertTokenizer.from_pretrained(self.args['tokenizer'])
        self.model = BERTDualHierarchicalTrsEncoder(
            model=self.args['pretrained_model'], 
            nlayer=self.args['nlayer'], 
            nhide=self.args['nhide'], 
            nhead=self.args['nhead'], 
            dropout=self.args['dropout']
        )
        self.load_checkpoint()
        if torch.cuda.is_available():
            self.model.cuda()
        self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
        
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_turn_length, cid_mask, rid_mask, recover_mapping = batch
            with autocast():
                loss, acc = self.model(
                    cid, rid, cid_turn_length, 
                    cid_mask, rid_mask, recover_mapping
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter):
        self.model.eval()
        pbar = tqdm(test_ite)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            cid, rids, cid_turn_length, cids_mask, rids_mask, label = batch
            batch_size = len(rids)
            scores = self.model.module.predict(cid, rids, cid_turn_length, cids_mask, rids_mask).cpu().tolist()    # [B]
            
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
        return (total_correct[0]/total_examples, total_correct[1]/total_examples, total_correct[2]/total_examples), avg_mrr, avg_prec_at_one, avg_map
