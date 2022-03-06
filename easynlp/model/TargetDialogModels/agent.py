from model.utils import *
from dataloader.util_func import *

class TargetDialogAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(TargetDialogAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
            self.log_save_file = open(path, 'w')
            if args['model'] in ['dual-bert-fusion']:
                self.inference = self.inference2
                print(f'[!] switch the inference function')
            elif args['model'] in ['dual-bert-one2many']:
                self.inference = self.inference_one2many
                print(f'[!] switch the inference function')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):

            # compatible with the curriculumn learning
            batch['mode'] = 'hard' if hard is True else 'easy'

            self.optimizer.zero_grad()

            if self.args['fgm']:
                with autocast():
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                self.fgm.attack()
                with autocast():
                    loss_adv, _ = self.model(batch)
                self.scaler.scale(loss_adv).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(
                    self.model.parameters(), 
                    self.args['grad_clip']
                )
                self.fgm.restore()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with autocast():
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # comment for the constant learning ratio
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num
   
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
            if 'context' in batch:
                cid, cid_mask = self.totensor([batch['context']], ctx=True)
                rid, rid_mask = self.totensor(batch['responses'], ctx=False)
                batch['ids'], batch['ids_mask'] = cid, cid_mask
                batch['rids'], batch['rids_mask'] = rid, rid_mask
            elif 'ids' in batch:
                if self.args['model'] in ['dual-bert-multi-ctx']:
                    pass
                elif self.args['model'] in ['dual-bert-session']:
                    pass
                else:
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

            # rerank by the compare model (bert-ft-compare)
            if rerank_agent:
                if 'context' in batch:
                    context = batch['context']
                    responses = batch['responses']
                elif 'ids' in batch:
                    context = self.convert_to_text(batch['ids'].squeeze(0), lang=self.args['lang'])
                    responses = [self.convert_to_text(res, lang=self.args['lang']) for res in batch['rids']]
                    context = [i.strip() for i in context.split('[SEP]')]
                packup = {
                    'context': context,
                    'responses': responses,
                    'scores': scores,
                }
                # only the scores has been update
                scores = rerank_agent.compare_reorder(packup)

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
    
    def load_model(self, path):
        # ========== common case ========== #
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            # context encoder checkpoint
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.ctx_encoder.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.ctx_encoder.model.load_state_dict(new_state_dict, strict=False)

            # response encoders checkpoint
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.can_encoder.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.can_encoder.load_state_dict(new_state_dict)
        else:
            # test and inference mode
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')
