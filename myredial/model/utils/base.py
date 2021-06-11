from .header import *

'''
Base Agent
'''

class RetrievalBaseAgent:

    def __init__(self):
        # NOTE: for torch.cuda.amp
        # self.scaler = GradScaler()
        pass

    def show_parameters(self, args):
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter):
        raise NotImplementedError

    def set_test_interval(self):
        self.args['test_step'] = [int(self.args['total_step']*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0

    def test_now(self, test_iter, recoder):
        # test in the training loop
        index = self.test_step_counter
        (r10_1, r10_2, r10_5), avg_mrr, avg_p1, avg_map = self.test_model(test_iter)
        self.model.train()    # reset the train mode
        recoder.add_scalar(f'train-test/R10@1', r10_1, index)
        recoder.add_scalar(f'train-test/R10@2', r10_2, index)
        recoder.add_scalar(f'train-test/R10@5', r10_5, index)
        recoder.add_scalar(f'train-test/MRR', avg_mrr, index)
        recoder.add_scalar(f'train-test/P@1', avg_p1, index)
        recoder.add_scalar(f'train-test/MAP', avg_map, index)
        self.test_step_counter += 1

    def load_checkpoint(self):
        if self.args['checkpoint']['is_load']:
            path = self.args['checkpoint']['path']
            self.load_bert_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{path}')

    def load_bert_model(self, path):
        raise NotImplementedError

    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        elif self.args['mode'] in ['inference']:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        else:
            # test doesn't need DDP
            pass

    def load_model(self, path):
        # for test and inference
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        try:
            self.model.module.load_state_dict(state_dict)
        except:
            self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

