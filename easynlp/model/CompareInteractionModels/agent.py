from model.utils import *
from dataloader.util_func import *

class CompareInteractionAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(CompareInteractionAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
            self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()

        if args['model'] in ['dual-bert-scm', 'dual-bert-scm-hn', 'dual-bert-scm-hn-mch', 'dual-bert-scm-hn-with-easy', 'dual-bert-scm-hn-dist', 'dual-bert-scm-hn-dm', 'dual-bert-scm-hn-topk', 'dual-bert-scm-compare', 'dual-bert-scm-sdl', 'dual-bert-scm-hn-pos', 'dual-bert-scm-hn-g', 'dual-bert-scm-hn-dm', 'dual-bert-scm-mutual']:
            if self.args['is_step_for_training']:
                self.train_model = self.train_model_step
            self.test_model = self.test_model_dual_bert
            self.test_model_horse_human = self.test_model_horse_human_dual_bert
        elif args['model'] in ['bert-ft-scm']:
            self.test_model = self.test_model_bert_ft
        elif args['model'] in ['bert-ft-compare']:
            # if self.args['is_step_for_training']:
            #     self.train_model = self.train_model_step_compare
            self.test_model = self.test_model_compare
            pass

        self.show_parameters(self.args)

    def train_model_step_compare(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        loss, acc = self.model(
            batch, 
            optimizer=self.optimizer, 
            scaler=self.scaler, 
            grad_clip=self.args['grad_clip'], 
            scheduler=self.scheduler,
        )
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunAcc', acc, current_step)
        pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}; acc: {round(acc, 4)}')
        pbar.update(1)
    
    def train_model_step(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            loss, acc = self.model(batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunAcc', acc, current_step)
        pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}; acc: {round(acc, 4)}')
        pbar.update(1)

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        total_acc, total_token_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
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

            batch_num += 1
            total_loss += loss.item()
            total_acc += acc

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num
        
    def train_model_(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        total_acc, total_token_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            if self.args['model'] in ['bert-ft-compare', 'bert-ft-compare-token']:
                loss, acc, inner_time = self.model(
                    batch, 
                    optimizer=self.optimizer, 
                    scaler=self.scaler, 
                    grad_clip=self.args['grad_clip'], 
                    scheduler=self.scheduler,
                )
                for i in range(batch_num, batch_num+inner_time):
                    if whole_batch_num + i in self.args['test_step']:
                        self.test_now(test_iter, recoder)
                        break
                batch_num += inner_time
                total_loss += loss.item()
                total_acc += acc
                acc /= inner_time
            else:
                with autocast():
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                batch_num += 1
                if whole_batch_num + batch_num in self.args['test_step']:
                    self.test_now(test_iter, recoder)

                total_loss += loss.item()
                total_acc += acc

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num

    @torch.no_grad()
    def test_model_dual_bert(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask

            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                scores = self.model.predict(batch).cpu().tolist()    # [B]

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
                np.array(label.cpu().tolist()),
                10)
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
        return {
            f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
            f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
            f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
            'MRR': round(100*avg_mrr, 2),
            'P@1': round(100*avg_prec_at_one, 2),
            'MAP': round(100*avg_map, 2),
        }

    @torch.no_grad()
    def test_model_horse_human_dual_bert(self, test_iter, print_output=False, rerank_agent=None):
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
    
    @torch.no_grad()
    def test_model_horse_human(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = []
        valid_num, whole_num = 0, 0
        for batch in pbar:
            label = np.array(batch['label'])
            packup = {
                'context': batch['context'],
                'responses': batch['responses'],
                'labels': batch['label'],
            }
            scores = self.fully_compare(packup)
            whole_num += 1
            if scores is None:
                continue
            else:
                valid_num += 1
            collection.append((label, scores))
        print(f'[!] total sample: {whole_num}; valid sample: {valid_num}')
        return collection
            
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for batch in pbar:
            label = np.array(batch['label'])
            packup = {
                'context': batch['context'],
                'responses': batch['responses'],
                'labels': batch['label'],
            }
            if self.args['model'] in ['bert-ft-compare-token']:
                scores = self.fully_compare_token(packup)
            elif self.args['model'] in ['bert-ft-compare-multi', 'bert-ft-compare-multi-cls']:
                scores = self.fully_compare_multi(packup)
            else:
                scores = self.fully_compare(packup)
            # print output
            if print_output:
                c = batch['context']
                self.log_save_file.write(f'[Context] {c}\n')
                for r, score in zip(batch['responses'], scores):
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}] {r}\n')
                self.log_save_file.write('\n')

            rank_by_pred, pos_index, stack_scores = \
                calculate_candidates_ranking(
                    np.array(scores), 
                    np.array(label.tolist()),
                    10
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
            total_examples += math.ceil(label.size / 10)
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        return {
            f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
            f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
            f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
            'MRR': round(100*avg_mrr, 2),
            'P@1': round(100*avg_prec_at_one, 2),
            'MAP': round(100*avg_map, 2),
        }

    @torch.no_grad()
    def compare_one_turn(self, cids, rids, tickets, margin=0.0, soft=False):
        '''Each item pair in the tickets (i, j), the i has the bigger scores than j'''
        ids, tids, speaker_ids, recoder = [], [], [], []
        cpids = []
        for i, j in tickets:
            cids_, rids1, rids2 = deepcopy(cids), deepcopy(rids[i]), deepcopy(rids[j])
            truncate_pair_two_candidates(cids_, rids1, rids2, self.args['max_len'])
            ids_ = [self.cls] + cids_ + [self.sep] + rids1 + [self.sep] + rids2 + [self.sep]
            cpids_ = [0] * (len(cids_) + 2) + [1] * (len(rids1) + 1)  + [2] * (len(rids2) + 1)
            tids_ = [0] * (len(cids_) + 2) + [1] * (len(rids1) + 1) + [1] * (len(rids2) + 1)
            ids.append(ids_)
            cpids.append(cpids_)
            tids.append(tids_)
            recoder.append((i, j))
        ids = [torch.LongTensor(i) for i in ids]
        cpids = [torch.LongTensor(i) for i in cpids]
        tids = [torch.LongTensor(i) for i in tids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        cpids = pad_sequence(cpids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, cpids, mask = to_cuda(ids, tids, cpids, mask)
        # ===== make compare ===== # 
        batch_packup = {
            'ids': ids,
            'tids': tids,
            'cpids': cpids,
            'mask': mask,
        }
        if self.args['mode'] == 'train':
            comp_scores = self.model.module.predict(batch_packup)    # [B, 3]
        else:
            comp_scores = self.model.predict(batch_packup)    # [B, 3]
        comp_scores = comp_scores.tolist()
        
        if soft:
            return comp_scores, recoder

        # binary classification
        comp_label, n_recoder = [], []
        for s, (i, j) in zip(comp_scores, recoder):
            if s[1] >= margin + s[0]:
                comp_label.append(True)
                n_recoder.append((i, j))
            elif s[1] < s[0] - margin:
                comp_label.append(False)
                n_recoder.append((i, j))
        return comp_label, n_recoder

        # three classification
        # comp_label, n_recoder = [], []
        # for s, (i, j) in zip(comp_scores, recoder):
        #     if s[-1] >= self.args['ambiguous_margin']:
        #         # ambiguous cases
        #       pass
        #     elif s[1] >= margin + s[0]:
        #         comp_label.append(True)
        #         n_recoder.append((i, j))
        #     elif s[1] < margin - s[0]:
        #         comp_label.append(False)
        #         n_recoder.append((i, j))
        # return comp_label, n_recoder
    
    @torch.no_grad()
    def fully_compare_with_base(self, batch):
        self.model.eval() 
        ids = self.convert_text_to_ids(batch['context'])
        rids = self.convert_text_to_ids(batch['responses'])
        cids_ = self.convert_text_to_ids(batch['candidates'])
        cand_num = len(batch['candidates'])
        rids += cids_
        cids = []
        sids, cache = [], 0
        for u in cids_:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        sids.pop()
        cids.pop()
        tickets = []
        for i in range(10):
            # candidate idx
            for j in range(10, 10+cand_num):
                tickets.append((i, j))
        s, _ = self.compare_one_turn(cids, sids, rids, tickets, soft=True)
        scores = []
        for i in range(0, len(s), cand_num):
            scores.append(np.mean(s[i:i+cand_num]))
        return scores

    @torch.no_grad()
    def fully_comparae_p2(self, batch):
        self.model.eval() 
        pos_margin = self.args['positive_margin']
        items = self.convert_text_to_ids(batch['context'] + batch['responses'])
        cids_ = items[:len(batch['context'])]
        cids = []
        sids, cache = [], 0
        for u in cids_:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        sids.pop()
        cids.pop()
        rids = items[len(batch['context']):]
        tickets = []
        for i in range(len(rids)):
            for j in range(len(rids)):
                if i < j:
                    tickets.append((i, j))
        soft = False
        label, recoder = self.compare_one_turn(cids, sids, rids, tickets, margin=pos_margin, soft=soft)
        if soft is False:
            chain = {i: [] for i in range(len(rids))}
            # key is bigger than values
            for l, (i, j) in zip(label, recoder):
                if l is True:
                    chain[i].append(j)
                else:
                    chain[j].append(i)
            # scores = self.generate_scores_pagerank(chain)
            scores = self.generate_scores_counter(chain)
        else:
            # propagation scorer
            chain = torch.zeros(len(rids), len(rids))
            for l, (i, j) in zip(label, recoder):
                # advantage from i to j
                chain[i, j] = l
                chain[j, i] = 1-l
            scores = self.generate_scores_propagate_with_edge_weight(chain)
        return scores

    @torch.no_grad()
    def fully_compare(self, batch):
        self.model.eval() 
        pos_margin = self.args['positive_margin']
        items = self.convert_text_to_ids(batch['context'] + batch['responses'])
        cids_ = items[:len(batch['context'])]
        cids = []
        for u in cids_:
            cids.extend(u + [self.eos])
        cids.pop()
        rids = items[len(batch['context']):]
        tickets = []
        for i in range(len(rids)):
            for j in range(len(rids)):
                if i != j:
                    tickets.append((i, j))
        soft = False
        # label, recoder = self.compare_one_turn(cids, sids, rids, tickets, margin=pos_margin, soft=soft)
        label, recoder = self.compare_one_turn(cids, rids, tickets, margin=pos_margin, soft=soft)
        if soft is False:
            chain = {i: [] for i in range(len(rids))}
            # key is bigger than values
            for l, (i, j) in zip(label, recoder):
                if l is True:
                    chain[i].append(j)
                else:
                    chain[j].append(i)
            # topological sort scorer
            # scores, valid = self.generate_scores(chain)
            # scores = self.generate_scores(chain)
            # return scores
            # if valid:
            #     return scores
            # else:
            #     return None
            # pagerank scorer
            # scores = self.generate_scores_pagerank(chain)
            scores = self.generate_scores_counter(chain)
        else:
            chain = {i: [] for i in range(len(rids))}
            for s, (i, j) in zip(label, recoder):
                chain[i].append(s[1])
                chain[j].append(s[0])
            scores = [np.mean(chain[i]) for i in range(len(chain))] 
            # propagation scorer
            # chain = torch.zeros(len(rids), len(rids))
            # for l, (i, j) in zip(label, recoder):
            #     # advantage from i to j
            #     chain[i, j] = l
            #     chain[j, i] = 1-l
            # scores = self.generate_scores_propagate_with_edge_weight(chain)
        return scores

    @torch.no_grad()
    def compare_evaluation(self, test_iter):
        rest = []
        for batch in test_iter:
            scores = self.fully_compare(batch)
            c = batch['context']
            r1, r2 = batch['responses']

            items = self.convert_text_to_ids(c + [r1, r2])['input_ids']
            cids_, rids = items[0], items[1:]
            cids = []
            for u in cids_:
                cids.extend(u + [self.eos])
            cids.pop()

            tickets = [(0, 1)]
            label = self.compare_one_turn(cids, rids, tickets, margin=0, fast=True)
            label = label.tolist()[0]
            s = round(label*100, 2)
            item = {'context': c, 'responses': (r1, r2), 'score': s}
            rest.append(item)
        return rest
    
    @torch.no_grad()
    def compare_reorder(self, batch):
        '''
        input: batch = {
            'context': 'text string of the multi-turn conversation context, [SEP] is used for cancatenation',
            'responses': ['candidate1', 'candidate2', ...],
            'scores': [s1, s2, ...],
        }
        output the updated scores for the batch, the order of the responses should not be changed, only the scores are changed.
        '''
        self.model.eval() 
        compare_turn_num = self.args['compare_turn_num']
        pos_margin = self.args['positive_margin']
        scores = batch['scores']
        length = len(batch['context'])
        items = self.convert_text_to_ids(batch['context'] + batch['responses'])
        cids_, rids = items[:length], items[length:]
        cids, sids, cache = [], [], 0
        for u in cids_:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        sids.pop()
        cids.pop()

        # sort the rids (decrease order)
        order = np.argsort(scores)[::-1].tolist()
        backup_map = {o:i for i, o in enumerate(order)}    # old:new
        rids = [rids[i] for i in order]
        scores = [scores[i] for i in order]

        # tickets to the comparsion function
        before_dict = {i:i-1 for i in range(len(rids))}
        for idx in range(compare_turn_num):
            tickets = []
            if idx == 0:
                for i in range(len(rids)):
                    if before_dict[i] != -1:
                        tickets.append((before_dict[i], i))
            else:
                # find conflict
                counter = [[] for _ in range(len(rids))]
                for i in range(len(rids)):
                    b = before_dict[i]
                    if b != -1:
                        counter[b].append(i)
                # collect confliction tickets
                for pair in counter:
                    if len(pair) == 2:
                        i, j = pair
                        if scores[i] < scores[j]:
                            tickets.append((j, i))
                        else:
                            tickets.append((i, j))
                    elif len(pair) > 2:
                        raise Exception()
            # abort
            if len(tickets) == 0:
                break

            label, recoder = self.compare_one_turn(cids, sids, rids, tickets, margin=pos_margin)
            d = {j:i for l, (i, j) in zip(label, recoder) if l is False}
            d = sorted(list(d.items()), key=lambda x:x[0])    # left to right
            for j, i in d:
                # put the j before i (plus the scores)
                s_j, s_i = scores[j], scores[i]
                # get before score
                if before_dict[i] == -1:
                    s_i_before = scores[i] + 2.
                else:
                    s_i_before = scores[before_dict[i]]
                delta = s_i_before - s_i
                delta_s = random.uniform(0, delta)
                scores[j] = s_i + delta_s    # bigger than s_i but lower than s_i_before
                # change the before dict
                before_dict[j] = before_dict[i]
                before_dict[i] = j

        # backup the scores
        scores = [scores[backup_map[i]] for i in range(len(order))]
        return scores

    def convert_text_to_ids(self, texts):
        items = self.vocab.batch_encode_plus(texts, add_special_tokens=False)['input_ids']
        return items

    def generate_scores_counter(self, chain):
        scores = {key: len(lists) for key, lists in chain.items()}
        return [scores[i] for i in range(len(scores))]

    def generate_scores_propagate_with_edge_weight(self, matrix):
        scores = torch.ones(len(matrix)).unsqueeze(-1) / len(matrix)
        for _ in range(20):
            scores = torch.matmul(matrix, scores)
            scores = F.softmax(scores, dim=0) 
        scores = scores.squeeze(-1)
        return scores.tolist()

    def generate_scores_pagerank(self, edges):
        # reverse the edges
        new_edges = []
        for i, item_list in edges.items():
            for j in item_list:
                new_edges.append((j, i))
        g = PageRank(len(edges), new_edges)
        s = g.iter()
        return s

    def generate_scores(self, edges):
        '''topological sort'''
        # len(edges) = the number of the vertices
        num = len(edges)
        g = GraphC(num)
        for i, item_list in edges.items():
            for j in item_list:
                g.addEdge(i, j)
        rest = g.topologicalSort()
        scores = list(reversed(range(num)))
        scores = [(i, j) for i, j in zip(rest, scores)]
        scores = sorted(scores, key=lambda x:x[0])
        scores = [j for i, j in scores]
        # return scores, g.valid
        return scores

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            if self.args['model'] in ['dual-bert-comp-hn', 'dual-bert-comp', 'dual-bert-compare', 'dual-bert-scm-hn', 'dual-bert-scm-hn-mch', 'dual-bert-scm-hn-with-easy', 'dual-bert-scm-hn-dist', 'dual-bert-scm-hn-dm', 'dual-bert-scm-hn-topk', 'dual-bert-scm-compare', 'dual-bert-scm-sdl', 'dual-bert-scm-hn-pos', 'dual-bert-scm-hn-g', 'dual-bert-scm-hn-dm', 'dual-bert-scm']:
                if self.args['model'] in ['dual-bert-scm-compare']:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        k = k.replace("model.bert.", "")
                        new_state_dict[k] = v
                    missing, unexcept = self.model.ctx_encoder.load_state_dict(new_state_dict, strict=False)
                else:
                    self.checkpointadapeter.init(
                        state_dict.keys() ,
                        self.model.ctx_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.ctx_encoder.model.load_state_dict(new_state_dict)
                
                self.checkpointadapeter.init(
                    state_dict.keys() ,
                    self.model.can_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.can_encoder.model.load_state_dict(new_state_dict)
            # elif self.args['model'] in ['dual-bert-scm']:
            #     ctx_new_state_dict, res_new_state_dict = OrderedDict(), OrderedDict()
            #     for k, v in state_dict.items():
            #         if k.startswith('can_encoder'):
            #             res_new_state_dict[k[12:]] = v
            #         elif k.startswith('ctx_encoder'):
            #             ctx_new_state_dict[k[12:]] = v
            #     self.model.can_encoder.load_state_dict(res_new_state_dict)
            #     self.model.ctx_encoder.load_state_dict(ctx_new_state_dict)
            elif self.args['model'] in ['bert-ft-compare']:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # k = f'bert.{k.replace("model.bert.", "")}'
                    k = f'roberta.{k.replace("model.roberta.", "")}'
                    new_state_dict[k] = v
                missing, unexcept = self.model.model.load_state_dict(new_state_dict, strict=False)
                print(f'[!] missing parameters: {missing}')
                print(f'[!] unexcept parameters: {unexcept}')
            elif self.args['model'] in ['bert-ft-scm']:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace("model.bert.", "")
                    new_state_dict[k] = v
                missing, unexcept = self.model.model.load_state_dict(new_state_dict, strict=False)
            elif self.args['model'] in ['bert-ft-compare-multi', 'bert-ft-compare-multi-cls', 'bert-ft-compare-multi-ens', 'bert-ft-compare-token']:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace("model.bert.", "")
                    new_state_dict[k] = v
                missing, unexcept = self.model.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
        print(f'[!] ========== load checkpoint over ==========')

    @torch.no_grad()
    def test_model_fg(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = {}
        for idx, batch in enumerate(pbar):                
            owner = batch['owner']
            label = np.array(batch['label'])
            packup = {
                'context': batch['context'],
                'responses': batch['responses'],
                'labels': batch['label'],
            }
            scores = self.fully_compare(packup)
            if owner in collection:
                collection[owner].append((label, scores))
            else:
                collection[owner] = [(label, scores)]
        return collection
    
    @torch.no_grad()
    def fully_compare_multi(self, batch):
        self.model.eval() 
        items = self.convert_text_to_ids(batch['context'] + batch['responses'])
        cids_ = items[:len(batch['context'])]
        cids = []
        sids, cache = [], 0
        for u in cids_:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        sids.pop()
        cids.pop()
        rids = items[len(batch['context']):]
        scores = self.compare_one_turn_multi(cids, sids, rids)
        return scores
    
    @torch.no_grad()
    def compare_one_turn_multi(self, cids, sids, rids):
        ctx_max_length, res_max_length = self.args['ctx_max_length'], self.args['res_max_length']
        # length limitation
        rids = [i[:(res_max_length-2)] for i in rids]
        cids = cids[-(ctx_max_length-2):]
        sids = sids[-(ctx_max_length-2):]

        cids_ = [self.cls] + cids + [self.sep]
        sids_ = [sids[0]] + sids + [sids[-1]]
        tids_ = [0] * (len(cids) + 2)
        lids_ = [-100] * (len(cids) + 2)
        other_speaker = 1 if sids[-1] == 0 else 0
        tcache = 1
        for idx, r in enumerate(rids):
            cids_ += [idx + 1] + r + [self.sep]
            sids_ += [other_speaker] * (len(r) + 2)
            tids_ += [tcache] * (len(r) + 2)
            lids_ += [0] + [-100] * (len(r) + 1)
            # tcache = 0 if tcache == 1 else 1
        assert len(cids_) == len(sids_) == len(tids_) == len(lids_)
        cids_ = torch.LongTensor(cids_).unsqueeze(0)
        sids_ = torch.LongTensor(sids_).unsqueeze(0)
        tids_ = torch.LongTensor(tids_).unsqueeze(0)
        lids_ = torch.LongTensor(lids_).unsqueeze(0)
        mask = generate_mask(cids_)
        cids_, sids_, tids_, lids_, mask = to_cuda(cids_, sids_, tids_, lids_, mask)
        batch_packup = {
            'cids': cids_,
            'sids': sids_,
            'tids': tids_,
            'lids': lids_,
            'mask': mask,
        }
        if self.args['mode'] == 'train':
            comp_scores = self.model.module.predict(batch_packup)
        else:
            comp_scores = self.model.predict(batch_packup)
        return comp_scores.tolist()
    
    @torch.no_grad()
    def fully_compare_token(self, batch):
        self.model.eval() 
        items = self.convert_text_to_ids(batch['context'] + batch['responses'])
        cids_ = items[:len(batch['context'])]
        cids = []
        sids, cache = [], 0
        for u in cids_:
            cids.extend(u + [self.eos])
            sids.extend([cache] * (len(u) + 1))
            cache = 1 if cache == 0 else 0
        sids.pop()
        cids.pop()
        rids = items[len(batch['context']):]
        tickets = []
        for i in range(len(rids)):
            for j in range(len(rids)):
                if i < j:
                    tickets.append((i, j))
        label, recoder = self.compare_one_turn_token(cids, sids, rids, tickets)
        # propagation scorer
        chain = torch.zeros(len(rids), len(rids))
        for l, (i, j) in zip(label, recoder):
            # advantage from i to j
            chain[i, j] = l
        scores = self.generate_scores_propagate_with_edge_weight(chain)
        return scores
    
    @torch.no_grad()
    def compare_one_turn_token(self, cids, sids, rids, tickets):
        '''Each item pair in the tickets (i, j), the i has the bigger scores than j'''
        ids, tids, speaker_ids, tlids, recoder = [], [], [], [], []
        other_speaker = 1 if sids[-1] == 0 else 0
        for i, j in tickets:
            cids_, sids_, rids1, rids2 = deepcopy(cids), deepcopy(sids), deepcopy(rids[i]), deepcopy(rids[j])
            truncate_pair_two_candidates(cids_, rids1, rids2, self.args['max_len'], sids=sids_)
            ids_ = [self.cls] + cids_ + [self.sep] + [1] + rids1 + [self.sep] + [2] + rids2 + [self.sep]
            sids_ = [sids_[0]] + sids_ + [sids[-1]] + [other_speaker] * (len(rids1) + len(rids2) + 4)
            tids_ = [0] * (len(cids_) + 2) + [1] * (len(rids1) + 2) + [0] * (len(rids2) + 2)
            tlids_ = [-100] * (len(cids_) + 2) + [1] + [-100] * (len(rids1) + 1) + [1] + [-100] * (len(rids2) + 1)
            assert len(tlids_) == len(ids_) == len(sids_) == len(tids_)
            ids.append(ids_)
            speaker_ids.append(sids_)
            tids.append(tids_)
            tlids.append(tlids_)
            recoder.append((i, j))
        ids = [torch.LongTensor(i) for i in ids]
        speaker_ids = [torch.LongTensor(i) for i in speaker_ids]
        tids = [torch.LongTensor(i) for i in tids]
        tlids = [torch.LongTensor(i) for i in tlids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        speaker_ids = pad_sequence(speaker_ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        tlids = pad_sequence(tlids, batch_first=True, padding_value=-100)
        mask = generate_mask(ids)
        ids, speaker_ids, tids, mask, tlids = to_cuda(ids, speaker_ids, tids, mask, tlids)
        # ===== make compare ===== # 
        batch_packup = {
            'ids': ids,
            'sids': speaker_ids,
            'tids': tids,
            'mask': mask,
            'tlids': tlids,
        }
        if self.args['mode'] == 'train':
            comp_scores, comp_scores_reverse = self.model.module.predict(batch_packup)    # [B]
        else:
            comp_scores, comp_scores_reverse = self.model.predict(batch_packup)    # [B]
        comp_scores = comp_scores.tolist()
        comp_scores += comp_scores_reverse.tolist()
        recoder += [(j, i) for i, j in recoder]
        return comp_scores, recoder
    
    @torch.no_grad()
    def test_model_dual_bert(self, test_iter, print_output=False, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask

            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                if core_time:
                    bt = time.time()
                scores = self.model.predict(batch).cpu().tolist()    # [B]
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
                    
                    sort_index = np.argsort(scores)[::-1]
                    for sidx in sort_index:
                        rid = batch['rids'][sidx]
                        score = scores[sidx]
                        l = label[sidx]
                        rtext = self.convert_to_text(rid)
                        score = round(score, 4)
                        self.log_save_file.write(f'[Score {score}; Label {l}] {rtext}\n')

                    # for rid, score, l in zip(batch['rids'], scores, label.tolist()):
                    #     rtext = self.convert_to_text(rid)
                    #     score = round(score, 4)
                    #     self.log_save_file.write(f'[Score {score}; Label {l}] {rtext}\n')
                self.log_save_file.write('\n')

            rank_by_pred, pos_index, stack_scores = \
            calculate_candidates_ranking(
                np.array(scores), 
                np.array(label.cpu().tolist()),
                10)
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
    def test_model_bert_ft(self, test_iter, print_output=False):
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
                np.array(label.cpu().tolist()),
                10)
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
        return {
            f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
            f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
            f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
            'MRR': round(100*avg_mrr, 2),
            'P@1': round(100*avg_prec_at_one, 2),
            'MAP': round(100*avg_map, 2),
        }

    def _packup(self, cids, rids1, rids2):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        truncate_pair_two_candidates(cids_, rids1_, rids2_, self.args['max_len'])
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        tids = [0] * (len(cids_) + 1) + [1] * (len(rids1_) + 1) + [0] * (len(rids2_) + 1)
        return ids, tids

    @torch.no_grad()
    def rerank(self, batches, inner_bsz=2048):
        self.model.eval()
        scores = []
        for batch in tqdm(batches):
            assert len(batch['candidates']) <= 50
            batch['responses'] = batch['candidates']
            s = self.fully_compare(batch)
            scores.append(s)
        return scores
    
    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        idx = 0
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)

            if len(texts) > size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
                )
                embds, texts = [], []
                idx += 1
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts),
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        ids, ids_mask = self.totensor(texts, ctx=True)
        vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def compare_candidates(self, context_list, candidate1, candidate2):
        self.model.eval()
        items = self.vocab.batch_encode_plus(context_list + [candidate1, candidate2], add_special_tokens=False)['input_ids']
        cids = []
        rids1, rids2 = items[-2], items[-1]
        for u in items[:-2]:
            cids.extend(u + [self.eos])
        cids.pop()
        ids, tids, cpids = [], [], []
        ids_, tids_, cpids_ = self._packup(cids, rids1, rids2)
        ids.append(ids_)
        tids.append(tids_)
        cpids.append(cpids_)
        ids_, tids_, cpids_ = self._packup(cids, rids2, rids1)
        ids.append(ids_)
        tids.append(tids_)
        cpids.append(cpids_)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        cpids = pad_sequence(cpids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, cpids, mask = to_cuda(ids, tids, cpids, mask)
        batch = {
            'ids': ids,
            'tids': tids,
            'cpids': cpids,
            'mask': mask
        }
        score = self.model.predict(batch)    # [2, 2]
        score = (score[0, 1] + score[1, 0])/2
        return score.item()
        
    def _packup(self, cids_, rids1_, rids2_):
        cids, rids1, rids2 = deepcopy(cids_), deepcopy(rids1_), deepcopy(rids2_)
        truncate_pair_two_candidates(cids, rids1, rids2, self.args['max_len'])
        ids = [self.cls] + cids + [self.sep] + rids1 + [self.sep] + rids2 + [self.sep]
        cpids = [0] * (2 + len(cids)) + [1] * (len(rids1) + 1) + [2] * (len(rids2) + 1)
        tids = [0] * (len(cids) + 2) + [1] * (len(rids1) + 1) + [1] * (len(rids2) + 1)
        return torch.LongTensor(ids), torch.LongTensor(tids), torch.LongTensor(cpids)

    @torch.no_grad()
    def test_model_compare(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            label = batch['label']

            if self.args['mode'] in ['train']:
                scores_ = self.model.module.predict(batch).cpu().tolist()    # [12]
            else:
                scores_ = self.model.predict(batch).cpu().tolist()    # [12]
            scores = []
            for i in range(0, len(scores_), 3):
                scores.append(np.mean(scores_[i:i+3]))
                # scores.append(scores_[i:i+3])
            # choice A
            # scores[0].extend([1 - scores_[3], 1 - scores_[6], 1 - scores_[9]])
            # choice B
            # scores[1].extend([1 - scores_[0], 1 - scores_[7], 1 - scores_[10]])
            # choice C
            # scores[2].extend([1 - scores_[1], 1 - scores_[4], 1 - scores_[11]])
            # choice D
            # scores[3].extend([1 - scores_[2], 1 - scores_[5], 1 - scores_[8]])
            # scores = [np.mean(i) for i in scores]

            # print output
            # if print_output:
            if 0:
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
                np.array(label.cpu().tolist()),
                4
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
        return {
            f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
            f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
            f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
            'MRR': round(100*avg_mrr, 2),
            'P@1': round(100*avg_prec_at_one, 2),
            'MAP': round(100*avg_map, 2),
        }


