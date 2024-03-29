from model.utils import *
from inference_utils import *


class GenerationAgent(GenerationBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(GenerationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.load_last_step = None
        
        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        if args['model'] in ['gpt2']:
            # path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}_{args["decoding_method"]}_{args["file_name"]}.txt'
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}_{args["decoding_method"]}_{args["file_name"]}_{args["beam_width"]}_{args["model_prediction_confidence"]}.txt'
        elif args['model'] in ['simrag']:
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{args["version"]}_bw_{args["beam_width"]}_alpha_{args["model_prediction_confidence"]}_beta_{args["beta"]}.txt'

            self.bert_tokenizer = BertTokenizer.from_pretrained(args['bert_pretrained_model'])
            self.gpt2_tokenizer = vocab
        else:
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
        self.log_save_file = open(path, 'w')

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

        if args['mode'] == 'train':
            self.set_test_interval()
            if args['model'] in ['simctg']:
                self.load_latest_checkpoint()
            else:
                self.load_checkpoint()

        if self.args['model'] in ['gpt2-cl']:
            self.train_model = self.train_model_cl
        elif self.args['model'] in ['gpt2-rerank']:
            self.train_model = self.train_model_rerank
        elif self.args['model'] in ['gpt2']:
            self.test_model = self.test_model_inference
        elif self.args['model'] in ['gpt2-contrastive-search', 'dialog-simctg', 'simctg']:
            self.train_model = self.train_model_contrastive_search
        elif self.args['model'] in ['dialog-eva']:
            self.train_model = self.train_model_dialog_eva
        elif self.args['model'] in ['dialog-plato-v1']:
            self.train_model = self.train_model_plato_v1
        elif self.args['model'] in ['copygeneration']:
            self.train_model = self.train_model_copygeneration
        elif self.args['model'] in ['gpt2-un', 'gpt2-un-seq']:
            self.train_model = self.train_model_un
        elif self.args['model'] in ['gpt2-original', 'gpt2-original-wt103', 'knn-lm'] and self.args['dataset'] in ['wikitext103', 'en_wiki', 'copygeneration_lawmt']:
            self.test_model = self.test_model_wikitext
            # return
            if self.args['model'] == 'knn-lm':
                # add the searcher
                faiss_searcher_args = deepcopy(self.args)
                faiss_searcher_args['model'] = 'knn-lm'
                config = load_config(faiss_searcher_args)
                faiss_searcher_args.update(config)
                faiss_searcher = Searcher(
                    faiss_searcher_args['index_type'],
                    dimension=faiss_searcher_args['dimension'],
                    nprobe=faiss_searcher_args['index_nprobe']
                )
                pretrained_model_name = faiss_searcher_args['pretrained_model'].replace('/', '_')
                model_name = faiss_searcher_args['model']
                faiss_searcher.load(
                    f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',        
                    f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',        
                )
                self.model.init_searcher(faiss_searcher)
                print(f'[!] knn-lm generator baseline init over ...')

    def build_offline_index(self, iter_):
        size = self.model.module.build_offline_index(iter_)
        print(f'[!] build offline index over, size is: {size}')

    def train_model_copygeneration(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()

        # debug
        # if current_step <= 3331:
        #     pbar.update(1)
        #     return 

        with autocast():
            (mle_loss, cl_loss, phrase_loss), (mle_acc, phrase_acc) = self.model(batch)
            cl_loss *= self.args['cl_loss_alpha']
            loss = mle_loss + cl_loss + phrase_loss
            loss /= self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), current_step)
            recoder.add_scalar(f'train/RunCLLoss', cl_loss.item(), current_step)
            recoder.add_scalar(f'train/RunPhraseLoss', phrase_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
            recoder.add_scalar(f'train/PhraseAcc', phrase_acc, current_step)
        pbar.set_description(f'[!] loss(mle|cl|phrase): {round(mle_loss.item(), 4)}|{round(cl_loss.item(), 4)}|{round(phrase_loss.item(), 4)}; acc(token|phrase): {round(mle_acc*100, 2)}|{round(phrase_acc*100, 2)}')
        pbar.update(1)
        return

    def train_model_plato_v1(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            mle_loss, rs_loss, bow_loss, token_acc = self.model(batch)
            loss = mle_loss + rs_loss + bow_loss
            loss /= self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), current_step)
            recoder.add_scalar(f'train/RunRSLoss', rs_loss.item(), current_step)
            recoder.add_scalar(f'train/RunBOWLoss', bow_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', token_acc, current_step)
        pbar.set_description(f'[!] loss(mle|rs|bow): {round(mle_loss.item(), 4)}|{round(rs_loss.item(), 4)}|{round(bow_loss.item(), 4)}; token_acc: {round(token_acc*100, 2)}')
        pbar.update(1)
        return

    def train_model_dialog_eva(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        with autocast():
            loss, mle_acc = self.model(batch)
        if loss.isnan().item() is False:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if recoder:
                recoder.add_scalar(f'train/RunMLELoss', loss.item(), current_step)
                recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        return

    @torch.no_grad()
    def test_model_ppl_contrastive_search(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.eval()
        ppl = self.model.module.calculate_ppl(batch['ids'], batch['ods'], batch['ids_mask'])
        return ppl
    
    def train_model_contrastive_search(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            mle_loss, mle_acc, cl_loss, valid_token_num = self.model(batch)
            cl_loss *= self.args['cl_loss_alpha']
            loss = mle_loss + cl_loss
            loss /= self.args['iter_to_accumulate']

        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), current_step)
            recoder.add_scalar(f'train/RunCLLoss', cl_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
            recoder.add_scalar(f'train/ValidTokenNum', valid_token_num, current_step)
        pbar.set_description(f'[!] loss(mle|cl): {round(mle_loss.item(), 4)}|{round(cl_loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}; valid_token_num: {valid_token_num}')
        pbar.update(1)
    
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
        with autocast():
            loss, token_acc = self.model(batch)
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
        pbar.set_description(f'[!] loss(mle): {round(loss.item(), 4)}; acc: {round(token_acc*100, 2)}')
        pbar.update(1)
        return loss, token_acc

    def train_model_un(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            # unlikelyhood training
            if self.args['model'] in ['gpt2-un-seq'] and current_step >= self.args['seq_un_begin_step']:
                batch['token_un'] = False
            else:
                batch['token_un'] = True
            mle_loss, un_loss, token_acc = self.model(batch)
            loss = mle_loss + un_loss
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
            recoder.add_scalar(f'train/RunMLELoss', mle_loss.item(), current_step)
            recoder.add_scalar(f'train/RunUNLoss', un_loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', token_acc, current_step)
        pbar.set_description(f'[!] loss(mle|un): {round(mle_loss.item(), 4)}|{round(un_loss.item(), 4)}; acc: {round(token_acc*100, 2)}')
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
            # gpt2 batch inference need the positional ids and the left padding mechanism
            if self.args['model'] == 'doctttttquery':
                if self.args['mode'] == 'train':
                    logits = self.model.module.predict(batch)
                else:
                    logits = self.model.predict(batch)
                ppl = 0.
            else:
                if self.args['mode'] == 'train':
                    # logits = self.model.module.predict(batch)
                    # test during training only meansure the ppl
                    ppl = self.model.module.calculate_ppl(
                        batch['ids'], 
                        batch['ids_mask'], 
                        batch['pos_ids'],
                        batch['ids_label']
                    )
                else:
                    logits = self.model.predict(batch)
                    # ppl = self.model.calculate_ppl(
                    #     batch['ids'], 
                    #     batch['ids_mask'], 
                    #     batch['pos_ids'],
                    #     batch['ids_label']
                    # )
                    ppl = 0.
            PPL.append(ppl)
            if print_output and self.args['mode'] == 'test':
                for c, r in zip(batch['ids'], logits):
                    if self.args['lang'] == 'en':
                        ctx = ' '.join([i for i in self.vocab.convert_ids_to_tokens(c) if i not in ['[CLS]', '[PAD]', '[SEP]', '<|endoftext|>']])
                        res = ' '.join([i for i in self.vocab.convert_ids_to_tokens(r) if i not in ['[CLS]', '[PAD]', '[SEP]', '<|endoftext|>']])
                    else:
                        # ctx = ''.join([i for i in self.vocab.convert_ids_to_tokens(c) if i not in ['[CLS]', '[PAD]', '[SEP]']])
                        # res = ''.join([i for i in self.vocab.convert_ids_to_tokens(r) if i not in ['[CLS]', '[PAD]', '[SEP]']])
                        ctx = ''.join([i for i in self.vocab.convert_ids_to_tokens(c) if i not in ['[PAD]']])
                        res = ''.join([i for i in self.vocab.convert_ids_to_tokens(r)])
                        if '[SEP]' in res:
                            res = res[:res.index('[SEP]')]
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
        # rest['Distinct-char-1'] = np.mean(distinct_char_1)
        # rest['Distinct-char-3'] = np.mean(distinct_char_3)
        # rest['Distinct-char-5'] = np.mean(distinct_char_5)
        # rest['Distinct-word-1'] = np.mean(distinct_word_1)
        # rest['Distinct-word-3'] = np.mean(distinct_word_3)
        # rest['Distinct-word-5'] = np.mean(distinct_word_5)
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
        if self.args['model'] in self.args['no_train_models'] and self.args['model'] != 'copygeneration':
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            try:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.module.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.module.model.load_state_dict(new_state_dict)
            except Exception as error:
                print(error)
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.model.load_state_dict(new_state_dict)
            print(f'[!] load model from {path}')
            return
        if self.args['model'] in ['gpt2']:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        elif self.args['model'] in ['simctg']:
            if self.args['mode'] == 'train':
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                step = state_dict['step']
                model_state_dict = state_dict['model_state_dict']
                optimizer_state_dict = state_dict['optimizer_state_dict']
                scheduler_state_dict = state_dict['scheduler_state_dict']

                self.load_last_step = step
                self.scheduler.load_state_dict(scheduler_state_dict)
                self.optimizer.load_state_dict(optimizer_state_dict)
                self.model.module.model.load_state_dict(model_state_dict)
            else:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                model_state_dict = state_dict['model_state_dict']
                self.model.model.load_state_dict(model_state_dict)
            print(f'[!] load the latest model from {path}')
        elif self.args['model'] in ['gpt2-original']:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        elif self.args['model'] in ['gpt2-unlikely']:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        elif self.args['model'] in ['copygeneration']:
            # ipdb.set_trace()
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            retrieval_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/phrase-copy/best_{pretrained_model_name}_{self.args["version"]}.pt'
            self.model.retriever.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu'))['model_state_dict'])
            # self.model.retriever.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu')))
            print(f'[!] load model from:\n - {retrieval_path}')

            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            retrieval_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/gpt2-original/best_{pretrained_model_name}_{self.args["version"]}.pt'
            # retrieval_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/gpt2-original-wt103/best_{pretrained_model_name}_{self.args["version"]}.pt'
            self.model.generator.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu')))
            print(f'\n - {retrieval_path}')

            # pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            # retrieval_path = f'{self.args["root_dir"]}/ckpt/en_wiki/gpt2-original/best_{pretrained_model_name}_{self.args["version"]}.pt'
            # self.model.en_wiki_generator.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu')))
            # print(f'\n - {retrieval_path}')

            # load the copygeenration model
            try:
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                retrieval_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/knn-lm/best_{pretrained_model_name}_{self.args["version"]}.pt'
                self.model.knn_lm_generator.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu')))
                print(f'\n - {retrieval_path}')

                # load the wikitext103 knn-lm baseline
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                retrieval_path = f'{self.args["root_dir"]}/ckpt/wikitext103/knn-lm/best_{pretrained_model_name}_{self.args["version"]}.pt'
                self.model.wikitext103_knn_lm_generator.load_state_dict(torch.load(retrieval_path, map_location=torch.device('cpu')))
                print(f'\n - {retrieval_path}')
            except:
                pass
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
                # self.checkpointadapeter.init(
                #     state_dict.keys(),
                #     self.model.state_dict().keys(),
                # )
                # new_state_dict = self.checkpointadapeter.convert(state_dict)
                # self.model.load_state_dict(new_state_dict)
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
            print(f'[!] load model from {path}')

    @torch.no_grad()
    def batch_generation_inference(self, inf_iter, size=100000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        results, context, response = [], [], []
        counter = 0
        for batch in pbar:
            rest = self.model.module.predict(batch)
            for i in range(0, len(rest), self.args['inference_num']):
                results.append(rest[i:i+self.args['inference_num']])
            context.extend(batch['context'])
            response.extend(batch['response'])
            if len(context) > size:
                assert len(context) == len(response) == len(results)
                torch.save(
                    (context, response, results),
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                context, response, results = [], [], []
                counter += 1
        assert len(context) == len(response) == len(results)
        torch.save(
            (context, response, results),
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
        )

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
            if type(rest[0]) == list:
                # diverse generation
                self.log_save_file.write(f'[Prefix     ] {batch["text"][0]}\n')
                for idx, r in enumerate(rest):
                    sep = '' if self.args['lang'] == 'en' else ''
                    r = sep.join([self.vocab.decode(token) for token in r])
                    self.log_save_file.write(f'[Generation {idx}] {r}\n')
                    self.log_save_file.flush()
                self.log_save_file.write('\n')
            else:
                for r, t in zip(rest, batch['text']):
                    sep = '' if self.args['lang'] == 'en' else ''
                    r = sep.join([self.vocab.decode(token) for token in r])
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

    @torch.no_grad()
    def generate_dialog(self, batches):
        '''work with the deploy/genetation_dialog.py; batch size must be 1'''
        self.model.eval()
        rests = []
        for batch in batches:
            generation = self.model.predict(batch['context'])
            rests.append(generation)
        return rests

    @torch.no_grad()
    def generate(self, batch):
        '''work with the deploy/genetation.py'''
        self.model.eval()
        sentences = [item['context'] for item in batch['segment_list']]
        if 'decoding_method' in batch:
            self.model.switch_decoding_method(batch['decoding_method'])
        else:
            self.model.switch_decoding_method(self.args['default_decoding_method'])
        if 'generation_num' not in batch:
            generation_num = self.args['default_generation_num']
        else:
            generation_num = batch['generation_num']
        self.model.test_max_len = batch['max_gen_len'] if 'max_gen_len' in batch else self.args['max_gen_len']
        if 'sampling_prefix_len' in batch:
            default_sampling_prefix_len = self.args['sampling_prefix_len']
            self.model.args['sampling_prefix_len'] = batch['sampling_prefix_len']
        rests = []
        for sub in range(0, len(sentences), self.args['inner_bsz']):
            bsz = len(sentences[sub:sub+self.args['inner_bsz']]) 
            # prepare the inputs: ids, ids_mask, pos_ids; make sure the left padding
            tokens = [self.vocab.encode(s, add_special_tokens=False)[-self.args['max_prefix_len']:] for s in sentences[sub:sub+self.args['inner_bsz']]]
            max_length = max([len(item) for item in tokens])
            tokens = [[self.model.pad] * (max_length - len(item)) + item for item in tokens]
            ids = torch.LongTensor(tokens)
            ids_mask = generate_mask(ids, pad_token_idx=self.model.pad)
            pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == self.model.pad, 0) 
            ids, ids_mask, pos_ids = to_cuda(ids, ids_mask, pos_ids)
            # expand for the top-k results
            ids = ids.unsqueeze(1).expand(-1, generation_num, -1).reshape(bsz*generation_num, -1)
            ids_mask = ids_mask.unsqueeze(1).expand(-1, generation_num, -1).reshape(bsz*generation_num, -1)
            pos_ids = pos_ids.unsqueeze(1).expand(-1, generation_num, -1).reshape(bsz*generation_num, -1)
            generations = self.model.predict({
                'ids': ids, 'ids_mask': ids_mask, 'pos_ids': pos_ids,
            })
            # convert the generations from the ids to tokens
            for i in range(0, len(generations), generation_num):
                sep = '' if self.args['lang'] == 'zh' else ' '
                rs = [sep.join(self.vocab.convert_ids_to_tokens(generations[i+j])) for j in range(generation_num)]
                new_rests = []
                for instance in rs:
                    if '[SEP]' in instance:
                        instance = instance[:instance.index('[SEP]')]
                    instance = instance.replace('[UNK]', '')
                    new_rests.append(instance)
                rests.append(new_rests)
        # fallback the smapling_prefix_len
        if 'sampling_prefix_len' in batch:
            self.model.args['sampling_prefix_len'] = default_sampling_prefix_len
        return rests

    @torch.no_grad()
    def rerank(self, batch, inner_bsz=1024):
        '''only work for the test_recall scripts for the ppl evaluation'''
        self.model.eval()
        subscores = []
        for idx in range(0, len(batch['candidates']), inner_bsz):
            candidates = batch['candidates'][idx:idx+inner_bsz]
            ids, ids_mask, pos_ids, ids_label = self.convert_to_ids_dialog(
                batch['ctext'], 
                candidates
            ) 
            batch['ids'] = ids
            batch['ids_mask'] = ids_mask
            batch['pos_ids'] = pos_ids
            batch['ids_label'] = ids_label
            subscores.append(self.model.calculate_ppl(ids, ids_mask, pos_ids, ids_label))
        return np.mean(subscores)

    def convert_to_ids_dialog(self, context, responses):
        items = self.vocab.batch_encode_plus([context] + responses, add_special_tokens=False)['input_ids']
        context, responses = items[0], items[1:]
        ids, labels = [], []
        for r in responses:
            ctx = deepcopy(context)
            truncate_pair(ctx, r, self.args['max_len'])
            tokens = ctx + [self.sep] + r
            label = [self.pad] * (len(ctx) + 1) + r
            ids.append(torch.LongTensor(tokens))
            labels.append(torch.LongTensor(label))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad)
        ids, ids_label = ids[:, :-1], labels[:, 1:]
        ids_mask = generate_mask(ids)
        pos_ids = (ids_mask.long().cumsum(-1) - 1).masked_fill(ids_mask == 0, 0)
        ids, ids_mask, pos_ids, ids_label = to_cuda(ids, ids_mask, pos_ids, ids_label)
        return ids, ids_mask, pos_ids, ids_label

    @torch.no_grad()
    def simrag_talk(self, context_list, retrieval_list=[], topk_topp=False, scorer=None, copy_token_num=0):
        self.model.eval()

        # process the context into the token ids for simrag model
        ## 1. gpt2 tokens
        tokens = self.gpt2_tokenizer.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
        gpt2_tokens = []
        for u in tokens:
            gpt2_tokens.extend(u + [self.gpt2_tokenizer.sep_token_id])
        gpt2_tokens = gpt2_tokens[-self.args['max_len']:]
        if topk_topp is False:
            # prepare
            ids = torch.LongTensor(gpt2_tokens).unsqueeze(0)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            batch = {
                'ids': ids,
                'ids_mask': mask,
                'rag_sentences': retrieval_list
            }
            assert scorer is not None
            logits = self.model.predict(batch, scorer, copy_token_num)
        else:
            ids = torch.LongTensor(gpt2_tokens).unsqueeze(0)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            batch = {
                'ids': ids,
                'ids_mask': mask,
                'rag_sentences': retrieval_list
            }
            logits = self.model.predict_topk_topp(batch, copy_token_num)
        res = self.gpt2_tokenizer.decode(logits[0])
        if '[SEP]' in res:
            res = res[:res.index('[SEP]')]
        res = ''.join(res.split())
        return res

    @torch.no_grad()
    def simrag_talk(self, context_list, retrieval_list=[], topk_topp=False, scorer=None, copy_token_num=0):
        self.model.eval()

        # process the context into the token ids for simrag model
        ## 1. gpt2 tokens
        tokens = self.gpt2_tokenizer.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
        gpt2_tokens = []
        for u in tokens:
            gpt2_tokens.extend(u + [self.gpt2_tokenizer.sep_token_id])
        gpt2_tokens = gpt2_tokens[-self.args['max_len']:]
        if topk_topp is False:
            # prepare
            ids = torch.LongTensor(gpt2_tokens).unsqueeze(0)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            batch = {
                'ids': ids,
                'ids_mask': mask,
                'rag_sentences': retrieval_list
            }
            assert scorer is not None
            logits = self.model.predict(batch, scorer, copy_token_num)
        else:
            ids = torch.LongTensor(gpt2_tokens).unsqueeze(0)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            batch = {
                'ids': ids,
                'ids_mask': mask,
                'rag_sentences': retrieval_list
            }
            logits = self.model.predict_topk_topp(batch, copy_token_num)
        res = self.gpt2_tokenizer.decode(logits[0])
        if '[SEP]' in res:
            res = res[:res.index('[SEP]')]
        res = ''.join(res.split())
        return res

    @torch.no_grad()
    def inference_copygeneration(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, metas, texts = [], [], []
        counter = 0
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['ids_mask']
            # phrase_pos = batch['phrase_pos']
            phrase_text = batch['phrase_text']
            rep = self.model.module.get_phrase_rep(ids, ids_mask, phrase_pos).cpu()
            embds.append(rep)
            texts.extend(phrase_text)
            # metas.extend(phrase_meta)
            if len(texts) > size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                counter += 1
                texts = []
                embds = []
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
            )

    @torch.no_grad()
    def test_model_wikitext(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            # string = self.model.greedy_search(batch).replace('\n', ' ').strip()
            string = self.model.topk_topp_search(batch).replace('\n', ' ').strip()
            res = batch['gt']
            ctx = self.vocab.decode(batch['ids'][0])
            self.log_save_file.write(f'[Prefix      ] {ctx}\n')
            self.log_save_file.write(f'[GroundTruth ] {res}\n')
            self.log_save_file.write(f'[Generation  ] {string}\n\n')
            self.log_save_file.flush()
        return {}

    @torch.no_grad()
    def inference_knnlm(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        counter = 0
        for batch in pbar:
            rep, target = self.model(batch)
            embds.append(rep)
            texts.extend(target)
            if len(texts) > size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                counter += 1
                texts = []
                embds = []
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
            )

    @torch.no_grad()
    def test_model_ppl(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        PPL = []
        for idx, batch in enumerate(pbar):
            ppl = self.model.calculate_ppl(
                batch['ids'], 
                batch['ids_mask'], 
            )
            PPL.append(ppl)
            pbar.set_description(f'[!] ppl: {round(np.mean(PPL), 4)}')
        return np.mean(PPL)

    def save_model_long(self, path, current_step):
        model_state_dict = self.model.module.model.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                'model_state_dict' : model_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'step': current_step
            }, 
            path
        )
        print(f'[!] save model into {path}')
 
    def load_latest_checkpoint(self):
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}'
        prefix_name = f'best_{pretrained_model_name}_{self.args["version"]}_'
        checkpoints = []
        for file in os.listdir(path):
            if prefix_name in file:
                version = file[len(prefix_name):].strip('.pt')
                version = int(version)
                checkpoints.append((file, version))
        if len(checkpoints) == 0:
            print(f'[!] do not find the latest model checkpoints')
            return
        checkpoints = sorted(checkpoints, key=lambda x:x[-1])
        latest_checkpoint, version = checkpoints[-1]
        latest_checkpoint = os.path.join(path, latest_checkpoint)
        self.load_model(latest_checkpoint)
        print(f'[!] train start from step: {version}')

