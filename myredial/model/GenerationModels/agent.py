from model.utils import *
from inference_utils import *


class GenerationAgent(GenerationBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(GenerationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        
        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()

        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        if args['model'] in ['gpt2']:
            # path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}_{args["decoding_method"]}_{args["file_name"]}.txt'
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}_{args["decoding_method"]}_{args["file_name"]}_{args["beam_width"]}_{args["model_prediction_confidence"]}.txt'
        else:
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
        self.log_save_file = open(path, 'w')

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

        if self.args['model'] in ['gpt2-cl']:
            self.train_model = self.train_model_cl
        elif self.args['model'] in ['gpt2-rerank']:
            self.train_model = self.train_model_rerank
        elif self.args['model'] in ['gpt2']:
            self.test_model = self.test_model_inference
        elif self.args['model'] in ['gpt2-contrastive-search']:
            self.train_model = self.train_model_contrastive_search

    def build_offline_index(self, iter_):
        size = self.model.module.build_offline_index(iter_)
        print(f'[!] build offline index over, size is: {size}')
    
    def train_model_contrastive_search(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            mle_loss, mle_acc, cl_loss = self.model(batch)
            loss = mle_loss + cl_loss
            loss /= self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), mle_acc)
            recoder.add_scalar(f'train/RunCLLoss', cl_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
        pbar.set_description(f'[!] loss(mle|cl): {round(mle_loss.item(), 4)}|{round(cl_loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        return
    
    def train_model_cl(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            (mle_loss, mle_acc), (doc_level_cl_loss, doc_level_cl_acc) = self.model(batch)
            alpha_ratio = min(self.args['max_ratio'], current_step / self.args['total_step'])
            loss = mle_loss + alpha_ratio * doc_level_cl_loss
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), mle_acc)
            recoder.add_scalar(f'train/RunDocCLLoss', doc_level_cl_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
            recoder.add_scalar(f'train/DocCLAcc', doc_level_cl_acc, current_step)
        pbar.set_description(f'[!] loss(mle|doc): {round(mle_loss.item(), 4)}|{round(doc_level_cl_loss.item(), 4)}; acc(mle|doc): {round(mle_acc*100, 2)}|{round(doc_level_cl_acc*100, 2)}')
        pbar.update(1)
    
    def train_model_rerank(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        # coarse-grained loss and mle loss
        with autocast():
            batch['mode'] = 'coarse-grained'
            lm_loss, cg_loss, token_acc = self.model(batch)
            loss = lm_loss + cg_loss
            # loss = lm_loss
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # fine-grained loss
        with autocast():
            batch['mode'] = 'fine-grained'
            fg_loss, fg_acc = self.model(batch)
        self.scaler.scale(fg_loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # update the scheduler
        self.scheduler.step()

        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunLMLoss', lm_loss.item(), current_step)
            recoder.add_scalar(f'train/RunCGLoss', cg_loss.item(), current_step)
            recoder.add_scalar(f'train/RunFGLoss', fg_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', token_acc, current_step)
            recoder.add_scalar(f'train/FGAcc', fg_acc, current_step)
        pbar.set_description(f'[!] loss(lm|cg|fg): {round(lm_loss.item(), 4)}|{round(cg_loss.item(), 4)}|{round(fg_loss.item(), 4)}; acc(lm|fg): {round(token_acc*100, 2)}|{round(fg_acc*100, 2)}')
        pbar.update(1)
        return loss, token_acc

    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        # with autocast():
        if self.args['model'] in ['gpt2-un-seq'] and current_step >= self.args['seq_un_begin_step']:
            batch['token_un'] = False
        else:
            batch['token_un'] = True
        loss, token_acc = self.model(batch)
        loss /= self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', token_acc, current_step)
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(token_acc*100, 2)}')
        pbar.update(1)
        return loss, token_acc

    @torch.no_grad()
    def test_model(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        PPL, rest = [], {}
        distinct_char_1, distinct_char_3, distinct_char_5 = [], [], []
        distinct_word_1, distinct_word_3, distinct_word_5 = [], [], []
        for idx, batch in enumerate(pbar):
            if self.args['mode'] == 'train':
                logits = self.model.module.predict(batch)
                ppl = self.model.module.calculate_ppl(batch['ids'], batch['ids_mask'], batch['ids_label'])
            else:
                logits = self.model.predict(batch)
                ppl = self.model.calculate_ppl(batch['ids'], batch['ids_mask'], batch['ids_label'])
            PPL.append(ppl)
            if print_output:
                for c, r in zip(batch['ids'], logits):
                    ctx = ''.join([i for i in self.vocab.convert_ids_to_tokens(c) if i not in ['[CLS]', '[PAD]', '[SEP]']])
                    res = ''.join([i for i in self.vocab.convert_ids_to_tokens(r) if i not in ['[CLS]', '[PAD]', '[SEP]']])
                    self.log_save_file.write(f'[Prefix     ] {ctx}\n')
                    self.log_save_file.write(f'[Generation ] {res}\n\n')
                    self.log_save_file.flush()
                    # distinct metric
                    distinct_char_1.append(distinct_sentence_level_char(res, n=1))
                    distinct_char_3.append(distinct_sentence_level_char(res, n=3))
                    distinct_char_5.append(distinct_sentence_level_char(res, n=5))
                    distinct_word_1.append(distinct_sentence_level_word(res, n=1))
                    distinct_word_3.append(distinct_sentence_level_word(res, n=3))
                    distinct_word_5.append(distinct_sentence_level_word(res, n=5))

        rest['PPL'] = np.mean(PPL)
        rest['Distinct-char-1'] = np.mean(distinct_char_1)
        rest['Distinct-char-3'] = np.mean(distinct_char_3)
        rest['Distinct-char-5'] = np.mean(distinct_char_5)
        rest['Distinct-word-1'] = np.mean(distinct_word_1)
        rest['Distinct-word-3'] = np.mean(distinct_word_3)
        rest['Distinct-word-5'] = np.mean(distinct_word_5)
        return rest
    
    @torch.no_grad()
    def test_model_compare(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        ppl_pos, ppl_neg, p, r, f, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, meteor = [], [], [], [], [], [], [], [], [], [], []
        results = []
        for idx, batch in enumerate(pbar):            
            if self.args['mode'] == 'train':
                logits = self.model.module.predict(batch)     # [B, S, V]
            else:
                logits = self.model.predict(batch)     # [B, S, V]
            # calculate ppl
            ppl_pos_ = self.model.calculate_ppl(batch['pos_ids'], batch['pos_ids_mask'], batch['pos_label'])
            ppl_neg_ = []
            for neg_ids, neg_ids_mask, neg_label in zip(batch['neg_ids'], batch['neg_ids_mask'], batch['neg_label']):
                ppl_neg__ = self.model.calculate_ppl(
                    neg_ids,
                    neg_ids_mask, 
                    neg_label
                )
                ppl_neg_.append(ppl_neg__)
            ppl_neg_ = np.mean(ppl_neg_)
            ppl_pos.append(ppl_pos_)
            ppl_neg.append(ppl_neg_)

            gen_texts = []
            for logit in logits:
                tokens = [i for i in self.vocab.convert_ids_to_tokens(logit) if i not in ['[PAD]', '[CLS]', '[SEP]']]
                gen_texts.append(''.join(tokens))
            if print_output:
                for prefix_t, pos_t, gen_t in zip(batch['text'], batch['pos_text'], gen_texts):
                    self.log_save_file.write(f'[Prefix     ] {prefix_t}\n')
                    self.log_save_file.write(f'[Positive   ] {pos_t}\n')
                    self.log_save_file.write(f'[Generation ] {gen_t}\n\n')
                self.log_save_file.flush()
            for gt_t, gen_t in zip(batch['text'], gen_texts):
                results.append((gt_t, gen_t))
        return self.obtain_automatic_evaluation(results)

    def obtain_automatic_evaluation(self, results):
        # calculate the evalution results
        inner_bsz = 64
        p, r, f, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, meteor = [], [], [], [], [], [], [], [], []
        for idx in tqdm(range(0, len(results), inner_bsz)):
            candidates = [i[1] for i in results[idx:idx+inner_bsz]]
            references = [i[0] for i in results[idx:idx+inner_bsz]]
            P, R, F = self.bertscorer.score(candidates, references)
            p.extend(P.tolist())
            r.extend(R.tolist())
            f.extend(F.tolist())

            for candidate, reference in zip(candidates, references):
                candidate, reference = ' '.join(list(candidate)), ' '.join(list(reference))
                rest = self.nlgeval.compute_individual_metrics(ref=[reference], hyp=candidate)
                bleu_1.append(rest['Bleu_1'])
                bleu_2.append(rest['Bleu_2'])
                bleu_3.append(rest['Bleu_3'])
                bleu_4.append(rest['Bleu_4'])
                rouge_l.append(rest['ROUGE_L'])
                meteor.append(rest['METEOR'])
        p_ = np.mean(p)
        r_ = np.mean(r)
        f_ = np.mean(f)
        b_1 = np.mean(bleu_1)
        b_2 = np.mean(bleu_2)
        b_3 = np.mean(bleu_3)
        b_4 = np.mean(bleu_4)
        r_l = np.mean(rouge_l)
        meteor_ = np.mean(meteor)
        return {
            'BLEU-1': b_1,
            'BLEU-2': b_2,
            'BLEU-3': b_3,
            'BLEU-4': b_4,
            'ROUGE-L': r_l,
            'METEOR': meteor_,
            'BERTScore-P': p_,
            'BERTScore-R': r_,
            'BERTScore-F': f_,
        }

    def load_model(self, path):
        if self.args['model'] in self.args['no_train_models']:
            return
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['model'] in ['gpt2']:
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        elif self.args['model'] in ['gpt2-unlikely']:
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        else:
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
                # self.model.load_state_dict(state_dict)
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.load_state_dict(new_state_dict)
            print(f'[!] load model from {path}')

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            embeddings, ts = self.model.module.get_cand(rid, rid_mask)    # [B, S, E]
            embds.append(embeddings.cpu())
            texts.extend(ts)
        embds = torch.cat(embds, dim=0).numpy()
        for counter, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
            )
    
    @torch.no_grad()
    def test_model_inference(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            rest = self.model.predict(batch)
            for r, t in zip(rest, batch['text']):
                r = ''.join(self.vocab.convert_ids_to_tokens(r))
                self.log_save_file.write(f'[Prefix     ] {t}\n')
                self.log_save_file.write(f'[Generation ] {r}\n\n')
                self.log_save_file.flush()
        return {}
    
    @torch.no_grad()
    def test_model_compare(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        ppl_pos, ppl_neg, p, r, f, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, meteor = [], [], [], [], [], [], [], [], [], [], []
        results = []
        for idx, batch in enumerate(pbar):            
            if self.args['mode'] == 'train':
                logits = self.model.module.predict(batch)     # [B, S, V]
