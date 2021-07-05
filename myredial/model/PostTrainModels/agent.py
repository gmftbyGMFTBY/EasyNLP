from model.utils import *

class PostTrainAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(PostTrainAgent, self).__init__()
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
        pass

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_mlm_loss, total_cls_loss = 0, 0
        total_cls_acc, total_mlm_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                mlm_loss, cls_loss, token_acc, cls_acc = self.model(batch)
                loss = mlm_loss + cls_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_cls_acc += cls_acc
            total_mlm_acc += token_acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/MLMLoss', total_mlm_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunMLMLoss', mlm_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', token_acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', total_mlm_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', cls_acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', total_cls_acc/batch_num, idx)
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/MLMLoss', total_mlm_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/TokenAcc', total_mlm_acc/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_cls_acc/batch_num, idx_)
