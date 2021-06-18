from model.utils import *

class RepresentationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(RepresentationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
            self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
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
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                loss, acc = self.model(batch)
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
    def test_model(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                scores = self.model.predict(batch).cpu().tolist()    # [B]

            # print output
            if print_output:
                ctext = self.convert_to_text(batch['ids'])
                self.log_save_file.write(f'[CTX] {ctext}\n')
                for rid, score in zip(batch['rids'], scores):
                    rtext = self.convert_to_text(rid)
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}] {rtext}\n')
                self.log_save_file.write('\n')

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

    @torch.no_grad()
    def inference(self, inf_iter):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']

            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
        embds = torch.cat(embds, dim=0).numpy()
        torch.save(
            (embds, texts), 
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["local_rank"]}.pt'
        )

    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask

    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        ids = [torch.LongTensor(self.vocab.encode(text)) for text in texts]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        ids_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, ids_mask = ids.cuda(), ids_mask.cuda()

        vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def encode_candidates(self, texts):
        self.model.eval()
        ids = [torch.LongTensor(self.vocab.encode(text)) for text in texts]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        ids_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, ids_mask = ids.cuda(), ids_mask.cuda()

        vectors = self.model.get_cand(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def rerank_one_sample(self, batch):
        self.model.eval()
        context_text = batch['context']
        candidates_text = batch['candidates']

        ids = torch.LongTensor(self.vocab.encode(context_text))
            
        rids = [torch.LongTensor(self.vocab.encode(text)) for text in candidates_text]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.vocab.pad_token_id)
        rids_mask = self.generate_mask(rids)
        if torch.cuda.is_available():
            ids, rids, rids_mask = ids.cuda(), rids.cuda(), rids_mask.cuda()

        # scores
        batch = {
            'ids': ids,
            'rids': rids,
            'rids_mask': rids_mask,
        }
        scores = self.model.predict(batch)
        return scores

    @torch.no_grad()
    def rerank(self, batches):
        self.model.eval()
        scores = []
        for batch in batches:
            scores.append(self.rerank_one_sample(batch).tolist())
        return scores

