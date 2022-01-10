from model.utils import *

class SemanticSimilarityAgent(SimCSEBaseAgent):

    def __init__(self, vocab, model, args):
        super(SemanticSimilarityAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model

        # special token [EOS]
        special_tokens_dict = {'eos_token': '[EOS]'}
        self.vocab.add_special_tokens(special_tokens_dict)

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

        if args['model'] in ['wz-simcse-seed']:
            self.train_model = self.train_model_seed

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
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
    def inference_wz_simcse(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        idx = 0
        for batch in pbar:
            ids = batch['ids']
            tids = batch['tids']
            ids_mask = batch['ids_mask']
            text = batch['text']
            res = self.model.module.get_embedding(ids, tids, ids_mask).cpu()
            embds.append(res)
            texts.extend(text)

            if len(texts) > size:
                # writer
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_wz_simcse_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
                )
                embds, texts = [], []
                idx += 1
        # save the last datasets
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_wz_simcse_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    def load_model(self, path):
        if self.args['mode'] == 'train':
            if self.args['model'] in ['wz-simcse', 'wz-simcse-seed']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.model.encoder.load_state_dict(state_dict)
                print(f'[!] wz-simcse loads pre-trained model from {path}')
            elif self.args['model'] in ['simcse']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.encoder.load_state_dict(new_state_dict)
                print(f'[!] simcse loads pre-trained model from {path}')
            else:
                pass
        else:
            # test or inference
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
            print(f'[!] Inference mode: simcse loads pre-trained model from {path}')
    
    def train_model_seed(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_cl_loss, total_de_loss = 0, 0
        total_de_acc, total_cl_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                cl_loss, de_loss, cl_acc, de_acc = self.model(batch)
                loss = cl_loss + de_loss * self.args['alpha_weight']
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_de_loss += de_loss.item()
            total_cl_loss += cl_loss.item()
            total_de_acc += de_acc
            total_cl_acc += cl_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
           
            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DecoderLoss', total_de_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunDecoderLoss', de_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/CLLoss', total_cl_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RuCLLoss', cl_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/CLAcc', total_cl_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunCLAcc', cl_acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DecoderTokenAcc', total_de_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DecoderRunTokenAcc', de_acc, idx)
            pbar.set_description(f'[!] loss(de|cl): {round(total_de_loss/batch_num, 2)}|{round(total_cl_loss/batch_num, 2)}; acc(de|cl): {round(100*total_de_acc/batch_num, 2)}|{round(100*total_cl_acc/batch_num, 2)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DecoderLoss', total_de_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/CLLoss', total_cl_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/CLAcc', total_cl_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DecoderTokenAcc', total_de_acc/batch_num, idx_)

        return batch_num

    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        output = self.vocab(texts, padding=True, max_length=self.args['max_len'], truncation=True, return_tensors='pt')
        ids, ids_mask, tids = output['input_ids'], output['attention_mask'], output['token_type_ids']
        ids, ids_mask, tids = to_cuda(ids, ids_mask, tids)
        vectors = self.model.get_embedding(ids, tids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        if self.args['model'] in ['simcse']:
            return
        label_list, sim_list = [], []
        for idx, batch in enumerate(pbar):                
            label_list.extend(batch['label'])
            ids, tids, ids_mask = batch['s1_ids'], batch['s1_tids'], batch['s1_ids_mask']
            ids_1, tids_1, ids_mask_1 = batch['s2_ids'], batch['s2_tids'], batch['s2_ids_mask']
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(
                    ids, tids, ids_mask,
                    ids_1, tids_1, ids_mask_1,    
                )    # [B]
            else:
                scores = self.model.predict(
                    ids, tids, ids_mask,
                    ids_1, tids_1, ids_mask_1,    
                )    # [B]
            sim_list.extend(scores)
        corrcoef = scipy.stats.spearmanr(label_list, sim_list).correlation
        return {
            'corrcoef': round(corrcoef, 4),
        }
