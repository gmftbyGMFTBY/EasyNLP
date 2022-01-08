from model.utils import *
from dataloader.util_func import *

class MutualTrainingAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(MutualTrainingAgent, self).__init__()
        self.args, self.vocab = args, vocab
        self.model = model

        # init some tokens ids
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.best_test = {'bi-encoder': None, 'cross-encoder': None}

        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        else:
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{args["version"]}.txt'
            self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.set_swap_interval()
        self.show_parameters(self.args)

    def set_swap_interval(self):
        self.args['swap_step'] = [int(self.args['total_step']*i) for i in np.arange(0, 1+self.args['swap_interval'], self.args['swap_interval'])]
        self.swap_step_counter = 0
        print(f'[!] swap interval steps: {self.args["swap_step"]}')

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        pbar = tqdm(train_iter)
        batch_num, total_loss  = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            with autocast():
                loss = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # update counter
            total_loss += loss.item()
            batch_num += 1

            if whole_batch_num + batch_num in self.args['swap_step']:
                self.model.module.swap_training_model()
            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
             
            pbar.set_description(f'[{self.model.module.training_model}] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        return batch_num
   
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, core_time=False, test_model_name=''):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch, test_model_name=test_model_name).cpu().tolist()    # [B]
            else:
                if core_time:
                    bt = time.time()
                scores = self.model.predict(batch, test_model_name=test_model_name).cpu().tolist()    # [B]
                if core_time:
                    et = time.time()
                    core_time_rest += et - bt

            # print output
            if print_output:
                if 'responses' in batch:
                    self.log_save_file.write(f'[CTX] {batch["context"]}\n')
                    for rtext, score in zip(responses, scores):
                        score = round(score, 4)
                        self.log_save_file.write(f'[Score {score}] {rtext}\n')
                else:
                    ctext = self.convert_to_text(batch['ids'].squeeze(0))
                    self.log_save_file.write(f'[CTX] {ctext}\n')
                    for rid, score in zip(batch['rids'], scores):
                        rtext = self.convert_to_text(rid)
                        score = round(score, 4)
                        self.log_save_file.write(f'[Score {score}] {rtext}\n')
                self.log_save_file.write('\n')

            rank_by_pred, pos_index, stack_scores = \
            calculate_candidates_ranking(
                np.array(scores), 
                np.array(label),
                10,
            )
            num_correct = logits_recall_at_k(pos_index, k_list)
            if self.args['dataset'] in ["douban", "restoration-200k"]:
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
        if core_time:
            return {
                f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
                f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
                f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
                'MRR': round(100*avg_mrr, 2),
                'P@1': round(100*avg_prec_at_one, 2),
                'MAP': round(100*avg_map, 2),
                'core_time': core_time_rest,
            }
        else:
            return {
                f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
                f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
                f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
                'MRR': round(100*avg_mrr, 2),
                'P@1': round(100*avg_prec_at_one, 2),
                'MAP': round(100*avg_map, 2),
            }
    
    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        if self.args['model'] in ['dual-bert-pos', 'dual-bert-hn-pos']:
            ids, ids_mask, pos_w = self.totensor(texts, ctx=True, position=True)
            vectors = self.model.get_ctx(ids, ids_mask, pos_w)    # [B, E]
        else:
            ids, ids_mask = self.totensor(texts, ctx=True)
            # vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
            vectors = self.model.module.get_ctx(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def encode_candidates(self, texts):
        self.model.eval()
        ids, ids_mask = self.totensor(texts, ctx=False)
        vectors = self.model.get_cand(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def rerank(self, batches, inner_bsz=2048):
        self.model.eval()
        scores = []
        for batch in batches:
            subscores = []
            # pbar = tqdm(range(0, len(batch['candidates']), inner_bsz))
            cid, cid_mask = self.totensor([batch['context']], ctx=True)
            # for idx in pbar:
            for idx in range(0, len(batch['candidates']), inner_bsz):
                candidates = batch['candidates'][idx:idx+inner_bsz]
                rid, rid_mask = self.totensor(candidates, ctx=False)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask
                batch['rids'] = rid
                batch['rids_mask'] = rid_mask
                subscores.extend(self.model.predict(batch).tolist())
            scores.append(subscores)
        return scores

    def load_checkpoint(self):
        self.load_model()

    def load_model(self):
        if self.args['mode'] == 'train':
            if self.args['model'] in ['trans-encoder']:
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["checkpoint"]["bi_encoder"]}'
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.checkpointadapeter.init(
                    state_dict.keys(), 
                    self.model.bi_encoder_model.state_dict().keys()
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.bi_encoder_model.load_state_dict(new_state_dict)

                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["checkpoint"]["cross_encoder"]}'
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.checkpointadapeter.init(
                    state_dict.keys(), 
                    self.model.cross_encoder_model.state_dict().keys()
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.cross_encoder_model.load_state_dict(new_state_dict)
        else:
            # test and inference mode
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model successfully')

    @torch.no_grad()
    def test_model_horse_human(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = []
        for batch in pbar:                
            ctext = '\t'.join(batch['ctext'])
            rtext = batch['rtext']
            label = batch['label']

            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask

            scores = self.model.predict(batch).cpu().tolist()

            # print output
            if print_output:
                self.log_save_file.write(f'[CTX] {ctext}\n')
                assert len(rtext) == len(scores)
                for r, score, l in zip(rtext, scores, label):
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}, Label {l}] {r}\n')
                self.log_save_file.write('\n')

            collection.append((label, scores))
        return collection

    def test_now(self, test_iter, recoder):
        index = self.test_step_counter
        self.test_one_model(test_iter, index, recoder, test_model_name='bi-encoder')
        # self.test_one_model(test_iter, index, recoder, test_model_name='cross-encoder')
        if self.model.module.training_model == 'bi-encoder':
            self.model.module.bi_encoder_model.train()
        else:
            self.model.module.cross_encoder_model.train()

    def test_one_model(self, test_iter, index, recoder, test_model_name=''):
        test_rest = self.test_model(test_iter, test_model_name=test_model_name)
        print(f'[{test_model_name}]:', test_rest)

        r10_1 = test_rest['R10@1']
        r10_2 = test_rest['R10@2']
        r10_5 = test_rest['R10@5']
        avg_mrr = test_rest['MRR']
        avg_p1 = test_rest['P@1']
        avg_map = test_rest['MAP']

        if recoder:
            recoder.add_scalar(f'train-test-{test_model_name}/R10@1', r10_1, index)
            recoder.add_scalar(f'train-test-{test_model_name}/R10@2', r10_2, index)
            recoder.add_scalar(f'train-test-{test_model_name}/R10@5', r10_5, index)
            recoder.add_scalar(f'train-test-{test_model_name}/MRR', avg_mrr, index)
            recoder.add_scalar(f'train-test-{test_model_name}/P@1', avg_p1, index)
            recoder.add_scalar(f'train-test-{test_model_name}/MAP', avg_map, index)
        self.test_step_counter += 1
        
        # find the new best model, save
        if self.args['local_rank'] == 0:
            # check the performance
            if self.compare_performance(test_rest, test_model_name=test_model_name):
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{test_model_name}_{self.args["version"]}.pt'
                self.save_model(save_path)
                print(f'[!] find new best {test_model_name} model at test step: {index}')

    def compare_performance(self, new_test, test_model_name=''):
        if self.best_test[test_model_name] is None:
            self.best_test[test_model_name] = new_test
            return True

        r10_1 = self.best_test[test_model_name]['R10@1']
        r10_2 = self.best_test[test_model_name]['R10@2']
        r10_5 = self.best_test[test_model_name]['R10@5']
        avg_mrr = self.best_test[test_model_name]['MRR']
        avg_p1 = self.best_test[test_model_name]['P@1']
        avg_map = self.best_test[test_model_name]['MAP']
        now_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 
        
        r10_1 = new_test['R10@1']
        r10_2 = new_test['R10@2']
        r10_5 = new_test['R10@5']
        avg_mrr = new_test['MRR']
        avg_p1 = new_test['P@1']
        avg_map = new_test['MAP']
        new_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 

        if new_test_score > now_test_score:
            self.best_test[test_model_name] = new_test
            return True
        else:
            return False
