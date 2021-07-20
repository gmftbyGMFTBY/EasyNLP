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
    
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_lm_loss, total_cls_loss = 0, 0, 0
        total_token_acc, total_cls_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                lm_loss, cls_loss, token_acc, cls_acc = self.model(batch)
                loss = cls_loss + lm_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_cls_loss += cls_loss.item()
            total_token_acc += token_acc
            total_cls_acc += cls_acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/LMLoss', total_lm_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLMLoss', lm_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_token_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', token_acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/CLSAcc', total_cls_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLSAcc', cls_acc, idx)
             
            pbar.set_description(f'[!] loss(lm/cls): {round(total_lm_loss/batch_num, 4)}|{round(total_cls_loss/batch_num, 4)}; acc(lm/cls): {round(total_token_acc/batch_num, 4)}|{round(total_cls_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/LMLoss', total_lm_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/TokenAcc', total_token_acc/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLSAcc', total_cls_acc/batch_num, idx_)
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            if self.args['mode'] in ['train']:
                acc = self.model.module.predict(batch)
            else:
                acc = self.model.predict(batch)
        return {
            f'R10@{k_list[0]}': 0,        
            f'R10@{k_list[1]}': 0,        
            f'R10@{k_list[2]}': 0,        
            'MRR': 0,
            'P@1': acc,
            'MAP': 0,
        }

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            # the context encoder model has been loaded (GPT-2)
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
