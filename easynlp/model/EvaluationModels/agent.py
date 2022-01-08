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
        from model.LanguageModels import GPT2LM
        self.lm_model = GPT2LM(**self.args) 
        self.lm_model.cuda()

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                output = self.model(batch)    # [B]
                label = batch['label']
                loss = self.criterion(output, label.to(torch.float))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            output = torch.sigmoid(output) > 0.5
            now_correct = torch.sum(output == label).item()
            correct += now_correct
            s += len(label)

            if batch_num in self.args['test_step']:
                if self.args['local_rank'] == 0:
                    pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                    save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                    self.save_model(save_path)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)
    
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
