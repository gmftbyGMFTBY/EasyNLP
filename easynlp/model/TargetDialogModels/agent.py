from model.utils import *
from dataloader.util_func import *
from inference_utils import *

class TargetDialogAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(TargetDialogAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
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
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
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
    def test_model(self, test_iter, print_output=False, rerank_agent=None, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        for idx, batch in enumerate(pbar):                
            label = batch['label']
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                if core_time:
                    bt = time.time()
                scores = self.model.predict(batch).cpu().tolist()    # [B]
                if core_time:
                    et = time.time()
                    core_time_rest += et - bt

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
    
    def load_model(self, path):
        # ========== common case ========== #
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            # context encoder checkpoint
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.ctx_encoder.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.ctx_encoder.model.load_state_dict(new_state_dict, strict=False)

            # response encoders checkpoint
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

    def add_cross_encoder(self, agent):
        self.cross_encoder_agent = agent

    # ===== init the memory of the topic ===== # 
    @torch.no_grad()
    def init(self, memory, topic, context_lists):
        # memorys: the list of the sentences containing the given topics
        self.topic = topic
        self.memory = memory

        # encode the memory by the response encoder
        ids = self.vocab.batch_encode_plus(memory, add_special_tokens=False)['input_ids']
        ids = [torch.LongTensor([self.cls] + i[:self.args['max_len']-2] + [self.sep]) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        self.memory_vector = self.model.get_cand(ids, ids_mask)    # [B, E]

        print(f'[!] init the memory and topic over')

        # load the ann searcher
        model_name = self.args['model']
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        self.searcher = Searcher(self.args['index_type'], dimension=self.args['dimension'], q_q=False, nprobe=self.args['index_nprobe'])
        faiss_ckpt_path = f'{self.args["root_dir"]}/data/{self.args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt'
        corpus_ckpt_path = f'{self.args["root_dir"]}/data/{self.args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'
        self.searcher.load(faiss_ckpt_path, corpus_ckpt_path)
        print(f'[!] init the ann searcher over')

        # init the cache
        self.cache = []
        self.cache_sequence = []
        print(f'[!] init the cache over')

        # init the representations of the context_lists
        # for utterance in context_lists:
        #     self.work([utterance])
        print(f'[!] init the given context lists over')

    @torch.no_grad()
    def work_no_topic(self, context_list):
        # 1. encode
        string = ' [SEP] '.join(context_list)
        ids = self.vocab.encode(string, add_special_tokens=False)
        ids = torch.LongTensor([self.cls] + ids[-self.args['max_len']+2:] + [self.sep])
        ids = ids.unsqueeze(0)
        ids_mask = torch.ones_like(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        
        # 2. obtian context representations
        cid_rep = self.model.get_ctx_embedding(ids, ids_mask)    # [B, E]
        # 3. search candidates
        candidates = self.searcher._search(cid_rep.cpu().numpy(), topk=self.args['work_topk'])[0]
        candidates = list(set(candidates) - set(context_list))
        return random.choice(candidates), -1

    @torch.no_grad()
    def work(self, context_list):
        # 0. get the candidate embeddings of utterances in context_list to avoid the repetition 
        ids = self.vocab.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
        ids = [torch.LongTensor([self.cls] + i[:self.args['max_len']-2] + [self.sep]) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        # past_rep = self.model.get_cand(ids, ids_mask)    # [S, E]
        past_rep = self.model.get_ctx_embedding(ids, ids_mask)    # [S, E]

        # 1. encode
        string = ' [SEP] '.join(context_list)
        ids = self.vocab.encode(string, add_special_tokens=False)
        ids = torch.LongTensor([self.cls] + ids[-self.args['max_len']+2:] + [self.sep])
        ids = ids.unsqueeze(0)
        ids_mask = torch.ones_like(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        
        # 2. obtian context representations
        cid_rep = self.model.get_ctx_embedding(ids, ids_mask)    # [B, E]

        # 3. search candidates
        candidates_ = self.searcher._search(cid_rep.cpu().numpy(), topk=self.args['work_topk'])[0]
        ## filter the candidates with past utterances
        candidates_ = list(set(candidates_) - set(context_list))

        # batches = [{'context': string, 'candidates': candidates_}]
        # scores = self.cross_encoder_agent.rerank(batches)[0]
        # ipdb.set_trace()

        # 4. encode the candidates by the context encoder for candidates rerank
        candidates = [string + ' [SEP] ' + candidate for candidate in candidates_]
        cids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        cids = [torch.LongTensor([self.cls] + i[-self.args['max_len']+2:] + [self.sep]) for i in cids]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        cids_mask = generate_mask(cids)
        cids, cids_mask = to_cuda(cids, cids_mask)
        cid_rep_ = self.model.get_ctx_embedding(cids, cids_mask)    # [B, E]

        # 6. response embedding
        # encode the memory by the response encoder
        ids = self.vocab.batch_encode_plus(candidates_, add_special_tokens=False)['input_ids']
        ids = [torch.LongTensor([self.cls] + i[:self.args['max_len']-2] + [self.sep]) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        cand_rep = self.model.get_cand(ids, ids_mask)    # [B, E]
        
        candidate_memory_score = torch.matmul(cand_rep, self.memory_vector.t()).max(dim=-1)[0]
        context_candidate_score = torch.matmul(cid_rep, cand_rep.t())[0]
        context_candidate_memory_score = torch.matmul(cid_rep_, self.memory_vector.t()).max(dim=-1)[0]    # [K]
        past_candidate_score = torch.matmul(cand_rep, past_rep.t()).max(dim=-1)[0]    # [B]

        # given the scores
        md = context_candidate_score + context_candidate_memory_score + candidate_memory_score - past_candidate_score

        dis, best = md.max(dim=-1)
        dis, best = dis.item(), best.item()    # distance range from -1 to 1

        # 6. return the best candidates and update the cache
        best_candidate = candidates_[best]
        return best_candidate, dis

    @torch.no_grad()
    def update_cache(self, tensor):
        self.cache.append(tensor)
        if self.cache_sequence:
            self.cache_sequence.extend([self.cache_sequence[-1] + 1] * len(tensor))
        else:
            self.cache_sequence.extend([0] * len(tensor))

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        counter = 0
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)

            if len(texts) >= size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                embds, texts = [], []
                counter += 1
        embds = torch.cat(embds, dim=0).numpy()
        torch.save(
            (embds, texts), 
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
        )
