from .header import *
from dataloader.util_func import *
from .utils import *

'''
Base Agent
'''

class RetrievalBaseAgent:

    def __init__(self):
        # open the test save scores file handler
        self.best_test = None 
        self.checkpointadapeter = CheckpointAdapter()

    def show_parameters(self, args):
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            if key in ['models', 'deploy', 'datasets', 'no_test_models', 'no_train_models']:
                # too long don't show
                continue
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
        print(f'[!] test interval steps: {self.args["test_step"]}')

    def compare_performance(self, new_test):
        if self.best_test is None:
            self.best_test = new_test
            return True

        r10_1 = self.best_test['R10@1']
        r10_2 = self.best_test['R10@2']
        r10_5 = self.best_test['R10@5']
        avg_mrr = self.best_test['MRR']
        avg_p1 = self.best_test['P@1']
        avg_map = self.best_test['MAP']
        now_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 
        
        r10_1 = new_test['R10@1']
        r10_2 = new_test['R10@2']
        r10_5 = new_test['R10@5']
        avg_mrr = new_test['MRR']
        avg_p1 = new_test['P@1']
        avg_map = new_test['MAP']
        new_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 

        if new_test_score > now_test_score:
            self.best_test = new_test
            return True
        else:
            return False

    def test_now(self, test_iter, recoder):
        index = self.test_step_counter
        test_rest = self.test_model(test_iter)
        r10_1 = test_rest['R10@1']
        r10_2 = test_rest['R10@2']
        r10_5 = test_rest['R10@5']
        avg_mrr = test_rest['MRR']
        avg_p1 = test_rest['P@1']
        avg_map = test_rest['MAP']

        recoder.add_scalar(f'train-test/R10@1', r10_1, index)
        recoder.add_scalar(f'train-test/R10@2', r10_2, index)
        recoder.add_scalar(f'train-test/R10@5', r10_5, index)
        recoder.add_scalar(f'train-test/MRR', avg_mrr, index)
        recoder.add_scalar(f'train-test/P@1', avg_p1, index)
        recoder.add_scalar(f'train-test/MAP', avg_map, index)
        self.test_step_counter += 1
        
        # find the new best model, save
        if self.args['local_rank'] == 0:
            # check the performance
            if self.compare_performance(test_rest):
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                self.save_model(save_path)
                print(f'[!] find new best model at test step: {index}')

        self.model.train()    # reset the train mode

    def load_checkpoint(self):
        if 'checkpoint' in self.args:
            if self.args['checkpoint']['is_load']:
                path = self.args['checkpoint']['path']
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{path}'
                self.load_model(path)
                print(f'[!] load checkpoint from {path}')
            else:
                print(f'[!] DONOT load checkpoint')
        else:
            print(f'[!] No checkpoint information found')

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
        self.checkpointadapeter.init(
            state_dict.keys(),
            self.model.state_dict().keys(),
        )
        new_state_dict = self.checkpointadapeter.convert(state_dict)
        self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    def convert_to_text(self, ids, lang='zh'):
        '''convert to text and ignore the padding token;
        no [CLS] and no the latest [SEP]'''
        if lang == 'zh':
            sep = '' if lang == 'zh' else ' '
            tokens = [self.vocab.convert_ids_to_tokens(i) for i in ids.cpu().tolist() if i != self.vocab.pad_token_id]
            text = sep.join(tokens)
        else:
            text = self.vocab.decode(ids).replace('[PAD]', '').strip()
        text = text.replace('[SEP]', ' [SEP] ').replace('[CLS]', '').replace('[UNK]', ' [UNK] ')
        text = text.strip(' [SEP] ')
        return text

    @torch.no_grad()
    def rerank(self, contexts, candidates):
        raise NotImplementedError

    def totensor(self, texts, ctx=True):
        items = self.vocab.batch_encode_plus(texts)['input_ids']
        if ctx:
            ids = [torch.LongTensor(length_limit(i, self.args['max_len'])) for i in items]
        else:
            ids = [torch.LongTensor(length_limit_res(i, self.args['res_max_len'], sep=self.sep)) for i in items]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, mask = to_cuda(ids, mask)
        return ids, mask

    def totensor_interaction(self, ctx_, responses_):
        '''for Interaction Models'''
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

        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        ids, tids = [], []
        for ctx, responses in zip(ctx_, responses_):
            ids_, tids_ = _encode_one_session(ctx, responses)
            ids.extend(ids_)
            tids.extend(tids_)

        ids = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids_, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
        return ids, tids, mask
