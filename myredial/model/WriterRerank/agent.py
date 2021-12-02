from model.utils import *
from dataloader.util_func import *
from .writer_gpt2 import *

class WriterRerankAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(WriterRerankAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
        self.log_save_file = open(path, 'w')
        
        self.gpt2_model = WriterGPT2Model(**args) 
        self.gpt2_vocab = self.gpt2_model.vocab
        
        if torch.cuda.is_available():
            self.model.cuda()
            self.gpt2_model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    @torch.no_grad()
    def gpt2_batch_inference(self, batch, inference_t, current_step):
        self.gpt2_model.eval()
        batch_size = len(batch['cids'])
        generation_rest = [[batch['rids'][i]] for i in range(batch_size)]
        generated = self.gpt2_model(batch, current_step)    # [B*inference_time]
        generated = [self.gpt2_vocab.decode(i) for i in generated]
        generated = self.vocab.batch_encode_plus(generated, add_special_tokens=False)['input_ids']
        counter = 0
        for idx in range(0, len(generated), inference_t):
            generation_rest[counter].extend(generated[idx:idx+inference_t])
            counter += 1

        batch['rids'] = generation_rest
        return batch

    def train_model(self, batch, recoder=None, current_step=0):
        self.model.train()
        # gpt2 batch inference
        batch = self.gpt2_batch_inference(batch, self.args['inference_time'], current_step)
        # discriminator
        self.optimizer.zero_grad()
        with autocast():
            loss, acc = self.model(batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/RunAcc', acc, current_step)
        return loss, acc
   
    @torch.no_grad()
    def test_model(self, test_iter, print_output=True):
        ##
        original_inference_t = test_iter.dataset.args['inference_time']
        test_iter.dataset.args['inference_time'] = 9
        ##
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        # R20@k
        k_list = [1, 2, 5, 10]
        core_time_rest = 0

        for idx, batch in enumerate(pbar):                
            # 1 + 9 + 10 = 20 candidates
            batch = self.gpt2_batch_inference(batch, 9, self.args['total_step'])
            label = torch.LongTensor([1] + [0] * 19)
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                bt = time.time()
                scores = self.model.predict(batch).cpu().tolist()    # [B]
                et = time.time()
                core_time_rest += et - bt

            ctext = ''.join(self.vocab.convert_ids_to_tokens(batch['cids'][0]))
            self.log_save_file.write(f'[Context] {ctext}\n')
            for rid, score in zip(batch['rids'][0] + batch['erids'], scores):
                rtext = ''.join(self.vocab.convert_ids_to_tokens(rid))
                score = round(score, 4)
                self.log_save_file.write(f'[Score {score}] {rtext}\n')
            self.log_save_file.write('\n')
            self.log_save_file.flush()

            rank_by_pred, pos_index, stack_scores = \
            calculate_candidates_ranking(
                np.array(scores), 
                np.array(label.cpu().tolist()),
                20
            )
            num_correct = logits_recall_at_k(pos_index, k_list)
            total_mrr += logits_mrr(pos_index)
            total_correct = np.add(total_correct, num_correct)
            total_examples += 1
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        ##
        test_iter.dataset.args['inference_time'] = original_inference_t
        ##
        return {
           f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
           f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
           f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
           'MRR': round(100*avg_mrr, 2),
           'P@1': round(100*avg_prec_at_one, 2),
           'MAP': round(100*avg_map, 2),
           'core_time': core_time_rest,
        }
