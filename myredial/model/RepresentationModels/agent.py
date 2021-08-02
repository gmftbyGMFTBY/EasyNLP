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
            elif self.args['model'] in ['dual-bert-ssl']:
                self.train_model = self.train_model_ssl
                # set hyperparameters
                self.model.ssl_interval_step = int(self.args['total_step'] * self.args['ssl_interval'])

            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
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
        # Metrics object
        self.metrics = Metrics()

    def train_model_hash(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_h_loss, total_q_loss, total_kl_loss = 0, 0, 0
        total_acc = 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                kl_loss, hash_loss, quantization_loss, acc = self.model(batch)
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
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/q_alpha', self.q_alpha, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/KLLoss', total_kl_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunKLLoss', kl_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] kl_loss: {round(kl_loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/KLLoss', total_kl_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HLoss', total_h_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
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
    
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            if self.args['model'] in ['dual-bert-gray-writer']:
                cid, cid_mask = self.totensor(batch['context'], ctx=True)
                rid, rid_mask = self.totensor(batch['responses'], ctx=False)
                batch['cid'], batch['cid_mask'] = cid, cid_mask
                batch['rid'], batch['rid_mask'] = rid, rid_mask

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
   
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            if 'context' in batch:
                cid, cid_mask = self.totensor([batch['context']], ctx=True)
                rid, rid_mask = self.totensor(batch['responses'], ctx=False)
                batch['ids'], batch['ids_mask'] = cid, cid_mask
                batch['rids'], batch['rids_mask'] = rid, rid_mask
            elif 'ids' in batch:
                cid = batch['ids'].unsqueeze(0)
                cid_mask = torch.ones_like(cid)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask

            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                scores = self.model.predict(batch).cpu().tolist()    # [B]

            # rerank by the compare model (bert-ft-compare)
            if rerank_agent:
                if 'context' in batch:
                    context = batch['context']
                    responses = batch['responses']
                elif 'ids' in batch:
                    context = self.convert_to_text(batch['ids'].squeeze(0), lang=self.args['lang'])
                    responses = [self.convert_to_text(res, lang=self.args['lang']) for res in batch['rids']]
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
            texts.extend(text * self.args['topk_encoder'])
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
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

    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        ids, ids_mask = self.totensor(texts, ctx=True)
        vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
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
        for batch in batches:
            subscores = []
            pbar = tqdm(range(0, len(batch['candidates']), inner_bsz))
            cid, cid_mask = self.totensor([batch['context']], ctx=True)
            for idx in pbar:
                candidates = batch['candidates'][idx:idx+inner_bsz]
                rid, rid_mask = self.totensor(candidates, ctx=False)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask
                batch['rids'] = rid
                batch['rids_mask'] = rid_mask
                subscores.extend(self.model.predict(batch).tolist())
            scores.append(subscores)
        return scores

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            if 'simsce' in path:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.load_state_dict(new_state_dict)
                print(f'[!] load the simcse pre-trained model')
            else:
                # context encoder checkpoint
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.ctx_encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_encoder.load_state_dict(new_state_dict)
                # response encoders checkpoint
                if self.args['model'] in ['dual-bert-grading', 'dual-bert-one2many-original']:
                    for i in range(self.args['topk_encoder']):
                        self.checkpointadapeter.init(
                            state_dict.keys(),
                            self.model.can_encoders[i].state_dict().keys(),
                        )
                        new_state_dict = self.checkpointadapeter.convert(state_dict)
                        self.model.can_encoders[i].load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-one2many']:
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
            # test and inference mode
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')
