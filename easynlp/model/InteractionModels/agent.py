from model.utils import *

class InteractionAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(InteractionAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model

        if self.args['model'] in ['bert-fp-original', 'bert-ft']:
            self.vocab.add_tokens(['[EOS]'])
            self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')

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

        if self.args['model'] in ['bert-ft-ibns']:
            self.train_model = self.train_ibns_model
        elif self.args['model'] in ['bert-ft-hier']:
            self.train_model = self.train_model_hier

        if self.args['is_step_for_training']:
            self.train_model = self.train_model_step

        self.criterion = nn.CrossEntropyLoss()
        self.show_parameters(self.args)
        
    def train_model_step(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            output = self.model(batch)
            loss = self.criterion(output, batch['label'])
            acc = (output.max(dim=-1)[1] == batch['label']).to(torch.float).mean().item()
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
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                if self.args['model'] in ['bert-ft-compare']:
                    output = self.model(batch)    # [B]
                    label = batch['label']
                    loss = self.criterion(output, label.to(torch.float))
                elif self.args['model'] in ['bert-ft-compare-plus']:
                    label = batch['label']
                    loss = self.model(batch)
                else:
                    # bert-ft
                    output = self.model(batch)    # [B]
                    label = batch['label']
                    loss = self.criterion(output, label)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if self.args['model'] in ['bert-ft-compare-plus']:
                output = output.max(dim=-1)[1]
                now_correct = (output == label).sum().item()
                s += len(label)
            else:
                now_correct = (output.max(dim=-1)[1] == label).sum().item()
                s += len(label)
            correct += now_correct

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return batch_num

    @torch.no_grad()
    def test_model_acc(self, test_iter, print_output=False, rerank_agent=None, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        acc = []
        for idx, batch in enumerate(pbar):
            label = batch['label']
            scores = F.softmax(self.model(batch), dim=-1)[:, 1]
            acc.append(((scores > 0.5) == label).to(torch.float).mean().item())
        acc = round(np.mean(acc), 4)
        print(f'[!] acc ratio: {acc}')
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        for idx, batch in enumerate(pbar):
            label = batch['label']
            if core_time:
                bt = time.time()
            scores = F.softmax(self.model(batch), dim=-1)[:, 1].cpu().tolist()
            if core_time:
                et = time.time()
                core_time_rest += et - bt

            if rerank_agent:
                scores_ = []
                counter = 0
                for i in tqdm(range(0, len(scores), 100)):
                    subscores = scores[i:i+100]
                    context = batch['context'][counter]
                    responses = batch['responses'][i:i+10]
                    packup = {
                        'context': context.split(' [SEP] '),
                        'responses': responses, 
                        'scores': subscores
                    }
                    subscores = rerank_agent.compare_reorder(packup)
                    scores_.append(subscores)
                    counter += 1
                scores = []
                for i in scores_:
                    scores.extend(i)
            
            # print output
            if print_output:
                for ids, score in zip(batch['ids'], scores):
                    text = self.convert_to_text(ids, lang=self.args['lang'])
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}] {text}\n')
                self.log_save_file.write('\n')
            
            rank_by_pred, pos_index, stack_scores = \
          calculate_candidates_ranking(
                np.array(scores), 
                np.array(label.cpu().tolist()),
                4)
            num_correct = logits_recall_at_k(pos_index, k_list)
            if self.args['dataset'] in ["douban", "restoration-200k"]:
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
    def rerank_and_return(self, batches, rank_num=64, keep_num=5, score_threshold=0.2, score_threshold_positive=0.7):
        self.model.eval()
        # hard negative selection
        '''
        ids, tids = [], []
        for batch in batches:
            context = batch['context']
            # select from the bad responses
            candidates = batch['candidates'][-rank_num:]
            assert len(candidates) == rank_num
            ids_, tids_ = self.make_tensor(context, candidates)
            ids.extend(ids_)
            tids.extend(tids_)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
        batch = {
            'ids': ids,
            'tids': tids,
            'ids_mask': mask
        }
        score = F.softmax(self.model(batch), dim=-1)[:, 1]
        min_indexes = []
        min_scores = []
        max_indexes, max_scores = [], []
        for i in range(0, len(score), rank_num):
            min_score, min_index = score[i:i+rank_num].topk(keep_num, largest=False)
            min_indexes.append(min_index.tolist())
            min_scores.append(min_score.tolist())
        '''

        # hard positive selection
        ids, tids = [], []
        for batch in batches:
            context = batch['context']
            candidates = batch['candidates'][:rank_num]
            assert len(candidates) == rank_num
            ids_, tids_ = self.make_tensor(context, candidates)
            ids.extend(ids_)
            tids.extend(tids_)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
        batch = {
            'ids': ids,
            'tids': tids,
            'ids_mask': mask
        }
        score = F.softmax(self.model(batch), dim=-1)[:, 1]
        max_indexes, max_scores = [], []
        for i in range(0, len(score), rank_num):
            max_score, max_index = score[i:i+rank_num].topk(keep_num)
            max_indexes.append(max_index.tolist())
            max_scores.append(max_score.tolist())
        assert len(max_indexes) == len(batches)

        #
        # hard_negative = []
        # for batch, score, index in zip(batches, min_scores, min_indexes):
        #     p = [(batch['candidates'][i], j) for i, j in zip(index, score) if j <= score_threshold]
        #     hard_negative.append(p)
        hard_positive = []
        for batch, score, index in zip(batches, max_scores, max_indexes):
            p = [(batch['candidates'][i], j) for i, j in zip(index, score) if j >= score_threshold_positive]
            hard_positive.append(p)
        # return hard_negative, hard_positive
        return hard_positive

    def make_tensor(self, ctx_, responses_):
        def _encode_one_session(ctx, responses):
            context_length = len(ctx)
            utterances = self.vocab.batch_encode_plus(ctx + responses, add_special_tokens=False)['input_ids']
            context_utterances = utterances[:context_length]
            response_utterances = utterances[context_length:]

            context = []
            for u in context_utterances:
                context.extend(u + [self.eos])
            context.pop()
    
            ids, tids = [], []
            for res in response_utterances:
                ctx = deepcopy(context)
                truncate_pair(ctx, res, self.args['max_len'])
                ids_ = [self.cls] + ctx + [self.sep] + res + [self.sep]
                tids_ = [0] * (len(ctx) + 2) + [1] * (len(res) + 1)
                ids.append(torch.LongTensor(ids_))
                tids.append(torch.LongTensor(tids_))
            return ids, tids
        ids, tids = _encode_one_session(ctx_, responses_)
        return ids, tids

    @torch.no_grad()
    def rerank(self, batches, inner_bsz=512):
        '''for bert-fp-original and bert-ft, the [EOS] token is used'''
        self.model.eval()
        scores = []
        for batch in batches:
            # collect ctx
            if type(batch['context']) == str:
                batch['context'] = [u.strip() for u in batch['context'].split('[SEP]')]
            elif type(batch['context']) == list:
                # perfect
                pass
            else:
                raise Exception()
            subscores = []
            pbar = tqdm(range(0, len(batch['candidates']), inner_bsz))
            for idx in pbar:
                candidates = batch['candidates'][idx:idx+inner_bsz]
                ids, tids, mask = self.totensor_interaction(batch['context'], candidates)
                batch['ids'], batch['tids'], batch['mask'] = ids, tids, mask
                subscores.extend(self.model(batch).tolist())
            scores.append(subscores)
        return scores
    
    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            if self.args['model'] in ['sa-bert']:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                missing, unexcept = self.model.model.load_state_dict(new_state_dict, strict=False)
            elif self.args['model'] in ['bert-ft-hier']:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.model.load_state_dict(new_state_dict)
            else:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.model.bert.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.model.bert.load_state_dict(new_state_dict)
            print(f'[!] ========= load model from {path}')
        else:
            # test and inference mode
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    @torch.no_grad()
    def test_model_fg(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = {}
        for idx, batch in enumerate(pbar):                
            owner = batch['owner']
            label = batch['label']
            scores = self.model(batch).cpu().tolist()    # [7]
            if owner in collection:
                collection[owner].append((label, scores))
            else:
                collection[owner] = [(label, scores)]
        return collection
    
    @torch.no_grad()
    def test_model_horse_human(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = []
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            scores = self.model(batch).cpu().tolist()    # [7]
            collection.append((label, scores))
        return collection
    
    def train_ibns_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_acc, total_loss, batch_num, correct, s = 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                score = self.model(batch)    # [B]

                score = torch.stack(torch.split(score, self.args['gray_cand_num']))
                mask = torch.zeros_like(score)
                mask[:, 0] = 1.
                loss_ = F.log_softmax(score, dim=-1) * mask
                loss = (-loss_.sum(dim=1)).mean()
                acc_num = (score.max(dim=-1)[1] == torch.zeros(len(score)).cuda()).sum().item()
                acc = acc_num / len(score)

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

    def train_model_hier(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                loss, acc = self.model(batch)    # [B]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            correct += acc
            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(correct/batch_num, 4)}')
        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', correct/batch_num, idx_)
        return batch_num

    @torch.no_grad()
    def inference_clean(self, inf_iter, inf_data, size=100000):
        self.model.eval()
        pbar = tqdm(inf_iter)

        results, writers = [], []
        for batch in pbar:
            if batch['ids'] is None:
                break
            raws = batch['raw']
            scores = self.model(batch)    # [B, 2]
            scores = F.softmax(scores, dim=-1)[:, 1].tolist()    # [B]
            for raw, s in zip(raws, scores):
                raw['bert_ft_score'] = round(s, 4)
            results.extend(raws)
            writers.extend(batch['writers'])
            if len(results) >= size:
                for rest, fw in zip(results, writers):
                    string = json.dumps(rest, ensure_ascii=False) + '\n'
                    fw.write(string)
                results, writers = [], []
        if len(results) > 0:
            for rest, fw in zip(results, writers):
                string = json.dumps(rest, ensure_ascii=False) + '\n'
                fw.write(string)
