from model.utils import *
from dataloader.util_func import *

class RepresentationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(RepresentationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            # hash-bert parameter setting
            if self.args['model'] in ['hash-bert']:
                self.q_alpha = self.args['q_alpha']
                self.q_alpha_step = (self.args['q_alpha_max'] - self.args['q_alpha']) / int(self.args['total_step'] / torch.distributed.get_world_size())
                self.train_model = self.train_model_hash
            elif self.args['model'] in ['bpr', 'hash-dual-bert-hier-trs']:
                self.train_model = self.train_model_bpr
            elif self.args['model'] in['lsh', 'lsh-hier']:
                self.train_model = self.train_model_lsh
            elif self.args['model'] in['sh']:
                self.train_model = self.train_model_sh
            elif self.args['model'] in['hash-bert-boost']:
                self.train_model = self.train_model_hash_boost
            elif self.args['model'] in['pq']:
                self.train_model = self.train_model_pq
            elif self.args['model'] in['itq']:
                self.train_model = self.train_model_itq
            elif self.args['model'] in ['dual-bert-ssl']:
                self.train_model = self.train_model_ssl
                # set hyperparameters
                self.model.ssl_interval_step = int(self.args['total_step'] * self.args['ssl_interval'])
            elif self.args['model'] in ['dual-bert-pt']:
                self.train_model = self.train_model_pt
            elif self.args['model'] in ['dual-bert-adv']:
                self.train_model = self.train_model_adv
            elif self.args['model'] in ['dual-bert-seed']:
                self.train_model = self.train_model_seed
            elif self.args['model'] in ['dual-bert-tacl', 'dual-bert-tacl-hn']:
                self.train_model = self.train_model_tacl
            elif self.args['model'] in ['phrase-copy']:
                if self.args['is_step_for_training']:
                    self.train_model = self.train_model_phrase_copy_step
                else:
                    self.train_model = self.train_model_phrase_copy

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
            pass
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    @torch.no_grad()
    def train_model_lsh(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        batch_num = 0
        for batch in tqdm(train_iter):
            if batch_num == 10:
                self.test_now(test_iter, recoder)
            batch_num += 1
        return batch_num

    @torch.no_grad()
    def train_model_itq(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):

        save_path = f'{self.args["root_dir"]}/data/{self.args["dataset"]}/hash_inference'
        if os.path.exists(f'{save_path}_0.pt'):
            reps = []
            try:
                for i in range(100):
                    path = f'{save_path}_{i}.pt'
                    reps_ = torch.load(path)
                    reps.append(reps_)
            except:
                pass
            if self.args['local_rank'] != 0:
                return batch_num
            reps = np.concatenate(reps)    # [B, E]
            batch_num = 0
        else:
            batch_num = 0
            reps = []
            for batch in tqdm(train_iter):
                rep = self.model(batch).cpu()
                if self.args['local_rank'] == 0:
                    reps.append(rep)
                batch_num += 1
            if self.args['local_rank'] != 0:
                return batch_num
            reps = torch.cat(reps).numpy()    # [B, E]
            counter = 0
            for i in range(0, len(reps), 500000):
                reps_ = reps[i:i+500000]
                torch.save(reps_, f"{save_path}_{counter}.pt")
                counter += 1
                print(f'[!] load subdataset size from {save_path}_{counter}.pt')
        if self.args['local_rank'] != 0:
            return batch_num
        print(f'[!] collect {len(reps)} samples for hash training')


        # begin to train the model
        ## 0. save the random seed
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])

        code_len = self.args['hash_code_size']
        R = torch.randn(code_len, code_len).cuda()
        U, _, _ = torch.svd(R)
        R = U[:, :code_len]
        ## 1. PCA
        pca = PCA(n_components=code_len)
        V = torch.from_numpy(pca.fit_transform(reps)).cuda()
        V = torch.tensor(V, dtype=torch.float32)
        ## 2. training
        for i in tqdm(range(self.args['max_itq_iter'])):
            V_tilde = V @ R
            B = V_tilde.sign()
            U, _, VT = torch.svd(B.t() @ V)
            R = (VT.t() @ U.t())

        ## save the necessary parameters: pca, R
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_itq_model_{self.args["version"]}.pt'
        torch.save((pca, R.cpu().numpy()), path)
        print(f'[!] save the itq model into {path}')
        return batch_num

    @torch.no_grad()
    def train_model_pq(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        save_path = f'{self.args["root_dir"]}/data/{self.args["dataset"]}/hash_inference'
        if os.path.exists(f'{save_path}_0.pt'):
            reps = []
            try:
                for i in range(100):
                    path = f'{save_path}_{i}.pt'
                    reps_ = torch.load(path)
                    reps.append(reps_)
                # creps = torch.cat(reps).numpy()    # [B, E]
            except:
                reps = np.concatenate(reps)    # [B, E]
            batch_num = 0
            if self.args['local_rank'] != 0:
                return batch_num
        else:
            batch_num = 0
            reps = []
            for batch in tqdm(train_iter):
                rep = self.model(batch).cpu()
                if self.args['local_rank'] == 0:
                    reps.append(rep)
                batch_num += 1
            if self.args['local_rank'] != 0:
                return batch_num
            reps = torch.cat(reps).numpy()    # [B, E]
            counter = 0
            for i in range(0, len(reps), 500000):
                reps_ = reps[i:i+500000]
                torch.save(reps_, f"{save_path}_{counter}.pt")
                counter += 1
        print(f'[!] collect {len(reps)} samples for pq training')

        # begin to train the model
        pq = nanopq.PQ(M=self.args['M'])
        pq.fit(reps)

        ## save the necessary parameters: pca, mn, R, modes
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_pq_model_{self.args["version"]}.pt'
        torch.save(pq, path)
        print(f'[!] save the sh model into {path}')
        return batch_num

    @torch.no_grad()
    def train_model_sh(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        save_path = f'{self.args["root_dir"]}/data/{self.args["dataset"]}/hash_inference'
        if os.path.exists(f'{save_path}_0.pt'):
            reps = []
            try:
                for i in range(100):
                    path = f'{save_path}_{i}.pt'
                    reps_ = torch.load(path)
                    reps.append(reps_)
                # creps = torch.cat(reps).numpy()    # [B, E]
            except:
                reps = np.concatenate(reps)    # [B, E]
            batch_num = 0
            if self.args['local_rank'] != 0:
                return batch_num
        else:
            batch_num = 0
            reps = []
            for batch in tqdm(train_iter):
                rep = self.model(batch).cpu()
                if self.args['local_rank'] == 0:
                    reps.append(rep)
                batch_num += 1
            if self.args['local_rank'] != 0:
                return batch_num
            reps = torch.cat(reps).numpy()    # [B, E]
            counter = 0
            for i in range(0, len(reps), 500000):
                reps_ = reps[i:i+500000]
                torch.save(reps_, f"{save_path}_{counter}.pt")
                counter += 1
        print(f'[!] collect {len(reps)} samples for hash training')

        # begin to train the model
        ## 0. save the random seed
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        ## 1. PCA
        pca = PCA(n_components=self.args['hash_code_size'])
        X = pca.fit_transform(reps)

        ## 2. fit uniform distribution
        eps = np.finfo(float).eps
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        ## 3. enumerate eigenfunctions
        R = mx - mn
        max_mode = np.ceil((self.args['hash_code_size'] + 1) * R / R.max()).astype(np.int)
        n_modes = max_mode.sum() - len(max_mode) + 1
        modes = np.ones([n_modes, self.args['hash_code_size']])
        m = 0
        for i in range(self.args['hash_code_size']):
            modes[m + 1: m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
            m = m + max_mode[i] - 1

        modes -= 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(n_modes, 0)
        eig_val = -(omegas ** 2).sum(1)
        ii = (-eig_val).argsort()
        modes = modes[ii[1:self.args['hash_code_size']+1], :]

        ## save the necessary parameters: pca, mn, R, modes
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_sh_model_{self.args["version"]}.pt'
        torch.save((pca, mn, R, modes), path)
        print(f'[!] save the sh model into {path}')
        return batch_num

    def train_model_bpr(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_acc, total_ref_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            batch['current_step'] = whole_batch_num + batch_num + 1
            with autocast():
                loss, acc, ref_acc, beta = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            total_ref_acc += ref_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] beta: {round(beta, 4)}; loss: {round(loss.item(), 4)}; acc(ref|hash): {round(total_ref_acc/batch_num, 4)}|{round(total_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num
    
    def train_model_hash_boost(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_h_loss, total_q_loss, total_kl_loss, total_dis_loss = 0, 0, 0, 0
        total_acc, total_ref_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
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
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(total_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num

    def train_model_hash(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_h_loss, total_q_loss, total_kl_loss, total_dis_loss = 0, 0, 0, 0
        total_acc, total_ref_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                batch['current_step'] = whole_batch_num + batch_num + 1
                kl_loss, hash_loss, quantization_loss, acc, ref_acc = self.model(batch)
                quantization_loss *= self.q_alpha
                loss = kl_loss + hash_loss + quantization_loss
                self.q_alpha += self.q_alpha_step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_q_loss += quantization_loss.item()
            total_h_loss += hash_loss.item()
            total_acc += acc
            total_ref_acc += ref_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/q_alpha', self.q_alpha, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] kl_loss: {round(kl_loss.item(), 4)}; q_loss: {round(quantization_loss.item(), 4)}; h_loss: {round(hash_loss.item(), 4)}; acc(ref|hash): {round(total_ref_acc/batch_num, 4)}|{round(total_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/QLoss', total_q_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/HLoss', total_h_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return batch_num
    
    def train_model_ssl(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
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

            # update may copy the parameters from original model to shadow model
            self.model.module.update()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    def train_model_pt(self, train_iter, test_iter, recoder=None, idx_=0):
        '''for dual-bert-pt model'''
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

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/MLMLoss', total_mlm_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunMLMLoss', mlm_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_mlm_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', token_acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_cls_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', cls_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(cls_loss.item(), 4)}|{round(total_cls_loss/batch_num, 4)}; acc: {round(cls_acc, 4)}|{round(total_cls_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/MLMLoss', total_mlm_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/TokenAcc', total_mlm_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_cls_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
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
            scores = self.model.predict_acc(batch).cpu().tolist()    # [B]
            scores = [1 if s > 0.5 else 0 for s in scores]
            acc_ = (torch.LongTensor(scores) == label.cpu()).to(torch.float).mean()
            acc.append(acc_.item())
        acc = round(np.mean(acc), 4)
        print(f'[!] acc: {acc}')
   
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
                if self.args['model'] in ['dual-bert-multi-ctx', 'dual-bert-session', 'dual-bert-hier-trs', 'dual-bert-mutual']:
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
    def inference_writer(self, inf_iter, size=1000000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        source = {}
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = [(t, ti) for t, ti in zip(batch['text'], batch['title'])]
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
            for t, u in zip(batch['title'], batch['url']):
                if t not in source:
                    source[t] = u
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

        # save sub-source
        torch.save(source, f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_subsource_{self.args["model"]}_{self.args["local_rank"]}.pt')
    
    @torch.no_grad()
    def inference_one2many(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text * (self.args['gray_cand_num']+1))
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_full_ctx_res(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        res_embds, ctx_embds, ctexts, rtexts = [], [], [], []
        counter = 0
        for batch in pbar:
            ids = batch['ids']
            rids = batch['rids']
            ids_mask = batch['ids_mask']
            rids_mask = batch['rids_mask']
            ctext = batch['ctext']
            rtext = batch['rtext']
            res = self.model.module.get_cand(rids, rids_mask).cpu()
            ctx = self.model.module.get_ctx(ids, ids_mask).cpu()
            res_embds.append(res)
            ctx_embds.append(ctx)
            ctexts.extend(ctext)
            rtexts.extend(rtext)

            if len(ctexts) > size:
                # save the memory
                res_embds = torch.cat(res_embds, dim=0).numpy()
                ctx_embds = torch.cat(ctx_embds, dim=0).numpy()
                torch.save(
                    (res_embds, ctx_embds, ctexts, rtexts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_ctx_res_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                res_embds, ctx_embds, ctexts, rtexts = [], [], [], []
                counter += 1
        if len(ctexts) > 0:
            res_embds = torch.cat(res_embds, dim=0).numpy()
            ctx_embds = torch.cat(ctx_embds, dim=0).numpy()
            torch.save(
                (res_embds, ctx_embds, ctexts, rtexts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_ctx_res_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
            )
    
    @torch.no_grad()
    def inference_with_source(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts, sources = [], [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            source = batch['ctext']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
            sources.extend(source)
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            source = sources[i:i+size]
            torch.save(
                (embd, text, source), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_with_source_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_data_filter(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        ctext, rtext, s = [], [], []
        for batch in pbar:
            ids = batch['ids']
            rids = batch['rids']
            ids_mask = batch['ids_mask']
            rids_mask = batch['rids_mask']
            cid_rep = self.model.module.get_ctx(ids, ids_mask)
            rid_rep = self.model.module.get_cand(rids, rids_mask)
            scores = (cid_rep * rid_rep).sum(dim=-1).tolist()    # [B]
            ctext.extend(batch['ctext'])
            rtext.extend(batch['rtext'])
            s.extend(scores)
        torch.save(
            (ctext, rtext, s), 
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_filter_{self.args["model"]}_{self.args["local_rank"]}.pt'
        )

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_context_for_response(self, inf_iter, size=1000000):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, responses = [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            response = batch['text']
            embd = self.model.module.get_ctx(ids, ids_mask).cpu()
            embds.append(embd)
            responses.extend(response)
        embds = torch.cat(embds, dim=0).numpy()
        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = responses[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_context_for_response_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

    @torch.no_grad()
    def inference_context_test(self, inf_iter, size=500000):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, contexts, responses = [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            context = batch['context']
            response = batch['responses']
            embd = self.model.module.get_ctx(ids, ids_mask).cpu()
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        embds = torch.cat(embds, dim=0).numpy()
        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            context = contexts[i:i+size]
            response = responses[i:i+size]
            torch.save(
                (embd, context, response), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_test_context_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_context(self, inf_iter, size=500000):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, contexts, responses = [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            context = batch['context']
            response = batch['response']
            embd = self.model.module.get_ctx(ids, ids_mask).cpu()
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        embds = torch.cat(embds, dim=0).numpy()
        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            context = contexts[i:i+size]
            response = responses[i:i+size]
            torch.save(
                (embd, context, response), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_context_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference2(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            cid = batch['cid']
            cid_mask = batch['cid_mask']
            text = batch['text']
            res = self.model.module.get_cand(cid, cid_mask, rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

    def hier_totensor(self, texts):
        ids = []
        turn_length = []
        for text in texts:
            text = self.vocab.batch_encode_plus(text, add_special_tokens=False)['input_ids']
            text = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in text]
            text = [torch.LongTensor(i) for i in text[-self.args['max_turn_length']:]]
            ids.extend(text)
            turn_length.append(len(text))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return ids, ids_mask, turn_length

    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        if self.args['model'] in ['dual-bert-pos', 'dual-bert-hn-pos']:
            ids, ids_mask, pos_w = self.totensor(texts, ctx=True, position=True)
            vectors = self.model.get_ctx(ids, ids_mask, pos_w)    # [B, E]
        elif self.args['model'] in ['dual-bert-hier-trs', 'hash-dual-bert-hier-trs']:
            ids, ids_mask, turn_length = self.hier_totensor(texts)
            vectors = self.model.get_ctx(ids, ids_mask, turn_length)
        else:
            ids, ids_mask = self.totensor(texts, ctx=True)
            vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
            # vectors = self.model.module.get_ctx(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def encode_candidates(self, texts):
        self.model.eval()
        ids, ids_mask = self.totensor(texts, ctx=False)
        vectors = self.model.get_cand(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def rerank(self, batches, inner_bsz=2048):
        self.model.eval()
        scores = []
        for batch in tqdm(batches):
            subscores = []
            cid, cid_mask = self.totensor([batch['context']], ctx=True)
            for idx in range(0, len(batch['candidates']), inner_bsz):
                candidates = batch['candidates'][idx:idx+inner_bsz]
                rid, rid_mask = self.totensor(candidates, ctx=False)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask
                batch['rids'] = rid
                batch['rids_mask'] = rid_mask
                subscores.extend(self.model.predict(batch).tolist())
            scores.append(subscores)
        return scores

    @torch.no_grad()
    def rerank_recall_evaluation(self, batch, inner_bsz=2048):
        self.model.eval()
        subscores = []
        cid, cid_mask = self.totensor([batch['ctext']], ctx=True)
        for idx in range(0, len(batch['candidates']), inner_bsz):
            candidates = batch['candidates'][idx:idx+inner_bsz]
            rid, rid_mask = self.totensor(candidates, ctx=False)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask
            batch['rids'] = rid
            batch['rids_mask'] = rid_mask
            subscores.extend(self.model.predict(batch).tolist())
        return np.mean(subscores)

    def load_model(self, path):
        # ========== special case ========== #
        if self.args['mode'] == 'train' and self.args['model'] in ['dual-bert-sp']:
            context_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["checkpoint"]["context_encoder_path"]}'
            response_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["checkpoint"]["response_encoder_path"]}'
            # load context from bert-fp
            state_dict = torch.load(context_path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.ctx_encoder.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.ctx_encoder.model.load_state_dict(new_state_dict, strict=False)
            # load response from bert-fp
            state_dict = torch.load(response_path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.can_encoder.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.can_encoder.model.load_state_dict(new_state_dict, strict=False)
            print(f'[!] load following PLMs:\n - context BERT encoder: {context_path}\n - response BERT encoder: {response_path}')
            return 
        # ========== common case ========== #
        if self.args['mode'] == 'train':
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            if self.args['model'] in ['dual-bert-one2many']:
                new_ctx_state_dict = OrderedDict()
                new_res_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'ctx_encoder' in k:
                        new_k = k.replace('ctx_encoder.', '')
                        new_ctx_state_dict[new_k] = v
                    elif 'can_encoder' in k:
                        new_k = k.replace('can_encoder.', '')
                        new_res_state_dict[new_k] = v
                    else:
                        raise Exception()
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                for idx in range(self.args['gray_cand_num'] + 1):
                    self.model.can_encoders[idx].load_state_dict(new_res_state_dict)
                print(f'[!] init the context encoder and {self.args["gray_cand_num"]+1} response encoders')
                # print(f'[!] init the context encoder and response encoders')
            elif self.args['model'] in ['lsh-hier', 'hash-dual-bert-hier-trs', 'hash-bert-boost']:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.base_model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.base_model.load_state_dict(new_state_dict)
            elif self.args['model'] in ['dual-bert-hier-dist']:
                dr_bert_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["dr_bert_path"]}'
                state_dict = torch.load(dr_bert_path, map_location=torch.device('cpu'))
                self.model.dr_bert.load_state_dict(state_dict)
                dr_bert_v2_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["dr_bert_v2_path"]}'
                # state_dict = torch.load(dr_bert_v2_path, map_location=torch.device('cpu'))
                # self.model.dr_bert_v2.load_state_dict(state_dict)
                print(f'[!] load the model from:\n - {dr_bert_path}\n - {dr_bert_v2_path}')
            elif self.args['model'] in ['dual-bert-hn-ctx', 'dual-bert-cl']:
                new_ctx_state_dict = OrderedDict()
                new_res_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'ctx_encoder' in k:
                        new_k = k.replace('ctx_encoder.', '')
                        new_ctx_state_dict[new_k] = v
                    elif 'can_encoder' in k:
                        new_k = k.replace('can_encoder.', '')
                        new_res_state_dict[new_k] = v
                    else:
                        raise Exception()
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                self.model.can_encoder.load_state_dict(new_res_state_dict)
            elif self.args['model'] in ['dual-bert-pt']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                new_ctx_state_dict = OrderedDict()
                new_can_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'cls.seq_relationship' in k:
                        pass
                    elif 'bert.pooler' in k:
                        pass
                    else:
                        k = k.lstrip('model.')
                        new_ctx_state_dict[k] = v
                        new_can_state_dict[k] = v
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                self.model.can_encoder.load_state_dict(new_can_state_dict)
            elif self.args['model'] in ['dual-bert-hier-trs']:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.ctx_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_encoder.model.load_state_dict(new_state_dict)
                self.model.can_encoder.model.load_state_dict(new_state_dict)
                try:
                    self.model.can_encoder_momentum.model.load_state_dict(new_state_dict)
                    print(f'[!] ========== momentum candidate encoder found and load ==========')
                except:
                    print(f'[!] ========== momentum candidate encoder not found ==========')
            elif self.args['model'] in ['xmoco']:
                # fast or slow context encoder load
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.ctx_fast_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_fast_encoder.model.load_state_dict(new_state_dict, strict=False)
                self.model.ctx_slow_encoder.model.load_state_dict(new_state_dict, strict=False)
                # fast or slow response encoder load
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.can_fast_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.can_fast_encoder.model.load_state_dict(new_state_dict, strict=False)
                self.model.can_slow_encoder.model.load_state_dict(new_state_dict, strict=False)
            else:
                # context encoder checkpoint
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.ctx_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_encoder.model.load_state_dict(new_state_dict, strict=False)

                if self.args['model'] in ['dual-bert-speaker']:
                    self.model.ctx_encoder_1.model.load_state_dict(new_state_dict, strict=False)

               
                # response encoders checkpoint
                if self.args['model'] in ['dual-bert-multi', 'dual-bert-one2many-original']:
                    for i in range(self.args['gray_cand_num']):
                        self.checkpointadapeter.init(
                            state_dict.keys(),
                            self.model.can_encoders[i].state_dict().keys(),
                        )
                        new_state_dict = self.checkpointadapeter.convert(state_dict)
                        self.model.can_encoders[i].load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-grading']:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.model.load_state_dict(new_state_dict)
                    self.model.hard_can_encoder.model.load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-proj']:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.model.load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-seed']:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.model.load_state_dict(new_state_dict)
                else:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.load_state_dict(new_state_dict)
        else:
            if self.args['model'] in ['sh']:
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_sh_model_{self.args["version"]}.pt'
                self.model.pca, self.model.mn, self.model.R, self.model.modes = torch.load(path)
                print(f'[!] load the sh hashing model parameters from {path}')
            elif self.args['model'] in ['itq']:
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_itq_model_{self.args["version"]}.pt'
                self.model.pca, self.model.R = torch.load(path)
                print(f'[!] load the itq hashing model parameters from {path}')
            elif self.args['model'] in ['pq']:
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_pq_model_{self.args["version"]}.pt'
                self.model.pq = torch.load(path)
                print(f'[!] load pq model from {path}')
            elif self.args['model'] in ['phrase-copy']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.model.module.load_state_dict(state_dict)
            else:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
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
            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask
            scores = self.model.predict(batch).cpu().tolist()    # [7]
            # print output
            if print_output:
                ctext = self.convert_to_text(batch['ids'].squeeze(0))
                self.log_save_file.write(f'[CTX] {ctext}\n')
                for rid, score in zip(batch['rids'], scores):
                    rtext = self.convert_to_text(rid)
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}] {rtext}\n')
                self.log_save_file.write('\n')
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
    
    def train_model_adv(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_dc_acc = 0
        total_tloss, total_dc_loss = 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):
            # add the progress for adv training
            p = (whole_batch_num + batch_num) / self.args['total_step']
            l = 2. / (1. + np.exp(-10. * p)) - 1
            batch['l'] = l
            # 

            self.optimizer.zero_grad()
            with autocast():
                loss, dc_loss, acc, dc_acc = self.model(batch)
                tloss = loss + dc_loss
            self.scaler.scale(tloss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_tloss += tloss.item()
            total_loss += loss.item()
            total_dc_loss += dc_loss.item()
            total_acc += acc
            total_dc_acc += dc_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DCLoss', total_dc_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunDCLoss', dc_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TotalLoss', total_tloss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunTotalLoss', tloss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DCAcc', total_dc_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunDCAcc', dc_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc(acc|dc_acc): {round(total_acc/batch_num, 4)}|{round(total_dc_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DCLoss', total_dc_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/TotalLoss', total_tloss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DCAcc', total_dc_acc/batch_num, idx_)
        return batch_num

    @torch.no_grad()
    def inference_and_update_index(self, inf_iter, inner_size=500000):
        # inference the whole dataset, only one worker can be allowed to run the following steps
        self.inference(inf_iter, size=inner_size)
        torch.distributed.barrier()
        if self.args['local_rank'] != 0:
            return
        # load all of the saved samples
        embds, texts = [], []
        for i in tqdm(range(args['total_workers'])):
            for idx in range(100):
                try:
                    embd, text = torch.load(
                        f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                    )
                    print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt')
                except:
                    break
                embds.append(embd)
                texts.extend(text)
                already_added.append((i, idx))
            if len(embds) > 10000000:
                break
        embds = np.concatenate(embds) 

        # init the faiss searcher
        self.searcher = Searcher(
            args['index_type'], dimension=args['dimension']
        )
        searcher._build(embds, texts, speedup=True)
        print(f'[!] train the searcher over')
        
        # save the faiss searcher
        model_name = args['model']
        pretrained_model_name = args['pretrained_model']
        self.searcher.save(
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
        )
        print(f'[!] update faiss index over')
    
    def train_model_seed(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_tloss, total_bloss = 0, 0
        total_cid_de_acc, total_rid_de_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):

            # compatible with the curriculumn learning
            batch['mode'] = 'hard' if hard is True else 'easy'

            self.optimizer.zero_grad()

            with autocast():
                loss, acc, cid_de_acc, rid_de_acc = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # comment for the constant learning ratio
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            total_cid_de_acc += cid_de_acc
            total_rid_de_acc += rid_de_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/ContextDecoderTokenAcc', total_cid_de_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/ContextDecoderRunTokenAcc', cid_de_acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/ResponseDecoderTokenAcc', total_rid_de_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/ResponseDecoderRunTokenAcc', rid_de_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(total_loss/batch_num, 4)}; acc: {round(total_acc/batch_num, 4)}; c_token_acc: {round(total_cid_de_acc/batch_num, 4)}; r_token_acc: {round(total_rid_de_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/ContextDecoderTokenAcc', total_cid_de_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/ResponseDecoderTokenAcc', total_rid_de_acc/batch_num, idx_)
        return batch_num
    
    def train_model_tacl(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_loss1, total_loss2, total_loss3 = 0, 0, 0, 0
        total_acc1, total_acc2, total_acc3 = 0, 0, 0
        pbar = tqdm(train_iter)
        batch_num = 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                (loss1, loss2, loss3), (acc1, acc2, acc3) = self.model(batch)
                loss = loss1 + loss2 + loss3
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_acc1 += acc1
            total_acc2 += acc2
            total_acc3 += acc3
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss_sentence_level', total_loss1/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss_inner_sentence', total_loss2/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Loss_inner_pair', total_loss3/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc_sentence_level', total_acc1/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc_token_level_inner_sentence', total_acc2/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc_token_level_inner_pair', total_acc3/batch_num, idx)
             
            pbar.set_description(f'[!] loss: {round(total_loss/batch_num, 4)}; acc: {round(total_acc1/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Loss_sentence_level', total_loss1/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Loss_token_level_inner_sentence', total_loss2/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Loss_token_level_inner_pair', total_loss3/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc_sentence_level', total_acc1/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc_token_level_inner_sentence', total_acc2/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc_token_level_inner_pair', total_acc3/batch_num, idx_)
        return batch_num

    @torch.no_grad()
    def inference_context_one_sample(self, context_list):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        tokens = self.vocab.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
        ids = []
        for u in tokens:
            ids.extend(u + [self.sep])
        ids.pop()
        ids = [self.cls] + ids[-self.args['max_len']+2:] + [self.sep]
        ids = torch.LongTensor(ids).unsqueeze(0)
        ids_mask = torch.ones_like(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        embd = self.model.get_ctx(ids, ids_mask).cpu()
        return embd.cpu().numpy()

    @torch.no_grad()
    def inference_context_one_batch(self, context_lists):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        ids_all = []
        for context_list in context_lists:
            tokens = self.vocab.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
            ids = []
            for u in tokens:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = [self.cls] + ids[-self.args['max_len']+2:] + [self.sep]
            ids = torch.LongTensor(ids)
            ids_all.append(ids)
        ids = pad_sequence(ids_all, batch_first=True, padding_value=self.pad)
        ids_mask = torch.ones_like(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        bt = time.time()
        # embd = self.model.module.get_ctx(ids, ids_mask)
        embd = self.model.get_ctx(ids, ids_mask)
        t = time.time() - bt
        try:
            return embd.cpu().numpy(), t
        except:
            return embd, t
        
    @torch.no_grad()
    def inference_clean(self, inf_iter, inf_data, size=100000):
        self.model.eval()
        pbar = tqdm(inf_iter)

        results, writers = [], []
        for batch in pbar:
            if batch['ids'] is None:
                break
            raws = batch['raw']
            scores = self.model.module.score(batch).tolist()    # [B]
            for raw, s in zip(raws, scores):
                raw['dr_bert_score'] = round(s, 4)
            results.extend(raws)
            writers.extend(batch['writers'])
            if len(results) >= size:
                for rest, fw in zip(results, writers):
                    string = json.dumps(rest, ensure_ascii=False) + '\n'
                    fw.write(string)
                results = []
                writers = []
        if len(results) > 0:
            for rest, fw in zip(results, writers):
                string = json.dumps(rest, ensure_ascii=False) + '\n'
                fw.write(string)

    @torch.no_grad()
    def inference_phrases(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        phrases, reps = [], []
        idx = 0 
        counter = 0
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['ids_mask']
            pos = batch['pos']
            text = batch['text']
            p, t = self.model.module.get_phrase_rep(ids, ids_mask, pos, text)
            reps.append(p)
            phrases.extend(t)
            if len(phrases) >= size:
                reps = torch.cat(reps, dim=0).cpu().numpy()
                torch.save(
                    (reps, phrases), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
                )
                phrases, reps = [], []
                idx += 1
            counter += len(t)
            pbar.set_description(f'[!] collect {counter} phrases for worker {self.args["local_rank"]}')
        if len(phrases) > 0:
            reps = torch.cat(reps, dim=0).cpu().numpy()
            torch.save(
                (reps, phrases), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

    def train_model_phrase_copy(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_phrase_acc, total_token_acc = 0, 0, 0
        total_tloss, total_bloss = 0, 0
        total_token_loss, total_phrase_loss, total_cl_loss = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                phrase_loss, phrase_acc, token_loss, token_acc, cl_loss = self.model(batch)
                loss = phrase_loss + token_loss + cl_loss

                # phrase_loss, phrase_acc, cl_loss = self.model(batch)
                # loss = phrase_loss + cl_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_phrase_loss += phrase_loss.item()
            
            total_token_loss += token_loss.item() 
            total_cl_loss += cl_loss.item()
            
            total_token_acc += token_acc
            
            total_phrase_acc += phrase_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/PhraseLoss', total_phrase_loss/batch_num, idx)
                
                recoder.add_scalar(f'train-epoch-{idx_}/CLLoss', total_cl_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_token_acc/batch_num, idx)
                
                recoder.add_scalar(f'train-epoch-{idx_}/PhraseAcc', total_phrase_acc/batch_num, idx)
            pbar.set_description(f'[!] loss(phrase|token|cl): {round(total_phrase_loss/batch_num, 2)}|{round(total_token_loss/batch_num, 2)}|{round(total_cl_loss/batch_num, 2)}; acc(phrase|token): {round(total_phrase_acc/batch_num, 4)}|{round(total_token_acc/batch_num, 4)}')
            # pbar.set_description(f'[!] loss(phrase|cl): {round(total_phrase_loss/batch_num, 4)}|{round(total_cl_loss/batch_num, 4)}; acc: {round(total_phrase_acc/batch_num, 4)}')
        return batch_num

    def train_model_phrase_copy_step(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            batch['if_freeze_gpt2'] = False
            batch['if_freeze_bert'] = True
            oloss, phrase_acc, total_acc = self.model(batch)
            loss = oloss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        if recoder:
            recoder.add_scalar(f'train/Loss', oloss.item(), current_step)
            recoder.add_scalar(f'train/PhraseAcc', phrase_acc, current_step)
            recoder.add_scalar(f'train/TotalAcc', total_acc, current_step)
        pbar.set_description(f'[!] loss: {round(oloss.item(), 2)}; acc(phrase|total): {round(phrase_acc, 4)}|{round(total_acc, 4)}')

        # optimize bert 
        '''
        if (current_step + 1) % self.args['optimize_bert_time'] == 0:
            batch['if_freeze_gpt2'] = True
            batch['if_freeze_bert'] = False
            counter = 0
            for _ in tqdm(range(self.args['optimize_bert_step'])):
                batch['bert_chunk_index'] = list(range(counter, min(len(batch['dids']), counter+self.args['bert_chunk_size'])))
                oloss, _, _ = self.model(batch)
                loss = oloss / self.args['optimize_bert_step']
                self.scaler.scale(loss).backward()
                counter += self.args['bert_chunk_size']
                if counter >= len(batch['dids']):
                    counter = 0
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        '''

        pbar.update(1)
