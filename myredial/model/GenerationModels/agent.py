from model.utils import *

class GenerationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(GenerationAgent, self).__init__()
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
    
    def train_model(
        self, train_iter, test_iter, recoder=None, idx_=0
    ):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_cl_loss, total_gen_loss, total_gen_acc = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()


            with autocast():
                (cl_loss, gen_loss, loss), (acc, gen_acc) = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_cl_loss += cl_loss.item()
            total_gen_loss += gen_loss.item()
            total_acc += acc
            total_gen_acc += gen_acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/CLLoss', total_cl_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLLoss', cl_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/GenLoss', total_gen_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunGenLoss', gen_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/LMAcc', total_gen_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLMAcc', gen_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(gen_acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLLoss', total_cl_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/GenLoss', total_gen_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        recoder.add_scalar(f'train-whole/LMAcc', total_gen_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):                
            if self.args['mode'] in ['train']:
                output = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                output = self.model.predict(batch).cpu().tolist()    # [B]
            ctext = self.convert_to_text(batch['ids'])
            self.log_save_file.write(f'[Context] {ctext}\n')
            gt = self.convert_to_text(batch['rids'][0])
            self.log_save_file.write(f'[Ground-Truth] {gt}\n')
            text = self.convert_to_text(inf)
            self.log_save_file.write(f'[Pred] {text}\n')
            self.log_save_file.write('\n')
