from model.utils import *

class EvaluationAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(EvaluationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
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

        self.criterion = nn.BCEWithLogitsLoss()
        self.show_parameters(self.args)

        # LM agent
        # from model.LanguageModels.kenlm import KeNLM
        # from model.LanguageModels import GPT2LM
        # self.lm_model = GPT2LM(**self.args) 
        # self.lm_model.cuda()

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
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

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}; acc: {round(acc, 4)}')
        return 
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            scores = torch.sigmoid(self.model(batch)).cpu().tolist()
            lm_scores = self.lm_model.predict(batch)
            # print output
            if print_output:
                for ids, score, ppl in zip(batch['ids'], scores, lm_scores):
                    text = self.convert_to_text(ids, lang=self.args['lang'])
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}; PPL {ppl}] {text}\n')
                self.log_save_file.write('\n')
        return {}
    
    @torch.no_grad()
    def evaluate(self, test_iter):
        def _correlation(output, score):
            r_spearmanr, p_spearmanr = spearmanr(output, score)
            r_pearsonr, p_pearsonr = pearsonr(output, score)

            spearmanr_res = str(np.round(r_spearmanr, 3)) + ' (' + str(np.round(p_spearmanr, 3)) + ')'
            pearsonr_res = str(np.round(r_pearsonr, 3)) + ' (' + str(np.round(p_pearsonr, 3)) + ')'
            return [spearmanr_res, pearsonr_res]

        
        self.model.eval()
        pbar = tqdm(test_iter)
        human_annotations, automatic_scores = [], []

        for idx, batch in enumerate(pbar):                
            human_annotations.extend(batch['score'])
            score = self.model.predict(batch).cpu().tolist()
            # score = self.model.predict(batch)
            automatic_scores.extend(score)
        # pearson and spearman scores
        ipdb.set_trace()
        sp, pr = _correlation(automatic_scores, human_annotations)
        print(f'[!] pearsonr:', pr)
        print(f'[!] spearmanr:', sp)
