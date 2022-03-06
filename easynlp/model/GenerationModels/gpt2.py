from model.utils import *
from .gpt2_token_rerank import *
from .utils import *


class InferenceGPT2Model(nn.Module):

    '''only for inference testing, not trained'''

    def __init__(self, **args):
        super(InferenceGPT2Model, self).__init__()
        model = args['pretrained_model']
        if args['decoding_method'] in ['token_rerank_search']:
            self.model = GPT2TokenRerankModel(**args)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model)
        # pad token is 0
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        # self.vocab = BertTokenizerFast.from_pretrained(model)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.test_max_len = args['gen_max_len']
        self.test_max_ctx_len = args['gen_max_ctx_len']
        self.repetition_penalty = args['repetition_penalty']
        self.args = args
        self.switch_decoding_method(self.args['decoding_method'])

    def switch_decoding_method(self, method_name):
        if method_name == 'contrastive_search':
            self.predict = self.predict_contrastive_search
        elif method_name == 'token_rerank_search':
            self.predict = self.predict_token_rerank_search
        elif method_name == 'contrastive_beam_search':
            self.predict = self.predict_contrastive_beam_search
        elif method_name == 'contrastive_batch_search':
            self.predict = self.predict_contrastive_batch_search
        elif method_name == 'greedy_search':
            self.predict = self.predict_greedy_search
        elif method_name == 'beam_search':
            self.predict = self.predict_beam_search
        elif method_name == 'topk_topp_repetition_penalty_search':
            self.predict = self.predict_topk_topp_repetition_penalty
        elif method_name == 'topk_topp_repetition_penalty_fast_search':
            self.predict = self.predict_topk_topp_repetition_penalty_fast
        elif method_name == 'topk_topp_repetition_penalty_batch_fast_search':
            self.predict = self.predict_topk_topp_repetition_penalty_batch_fast
        elif method_name == 'topk_topp_search':
            self.predict = self.predict_topk_topp
        elif method_name == 'topk_search':
            self.predict = self.predict_topk
        elif method_name == 'topp_search':
            self.predict = self.predict_topp
        else:
            raise Exception(f'[!] cannot find the deocidng method: {method_name}')
        print(f'[!] switch model to {method_name}')

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl
    
    @torch.no_grad()
    def predict_contrastive_beam_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(self.args['contrastive_generation_num'])]

        past_key_values = None
        last_hidden_states = None
        logits = None
        queue, queue_scores = [], []
        delta = torch.arange(self.args['beam_width']).cuda() * self.args['beam_width'] 
        for step in range(self.test_max_len):
            ids, past_key_values, last_hidden_states, logits, queue, queue_scores = ContrastiveDecodingOneStepBeamSearch(
                self.model,
                ids,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                past_key_values,
                last_hidden_states,
                self.vocab,
                logits,
                step,
                self.args['contrastive_generation_num'],
                queue,
                queue_scores,
                self.args['limited_size'],
                delta,
            )
        return queue
    
    @torch.no_grad()
    def predict_contrastive_batch_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]

        past_key_values = None
        last_hidden_states = None
        logits = None
        sampling_prefix_len = self.args['sampling_prefix_len']
        for step in range(self.test_max_len):
            ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepBatch(
                self.model,
                ids,
                ids_mask,
                ids_pos,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.sep,
                min(1., (step+1)/self.args['sep_smooth_length']),
                past_key_values,
                last_hidden_states,
                self.vocab,
                logits,
                step,
                step < self.args['sampling_prefix_len'],
            )
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            ids_mask = torch.ones_like(ids)
            # collect ids: [B, 1]
            tokens = ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
        # batch size is 1
        return generated

    @torch.no_grad()
    def predict_token_rerank_search(self, batch):
        self.model.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        for step in range(self.test_max_len):
            input_ids = TokenRerankDecodingOneStep(
                self.model.model,
                self.model.token_reranker,
                input_ids,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.args['sampling_probability'],
                self.sep,
                min(1., (step+1)/self.args['sep_smooth_length']),
            )
        # input_ids contains the prefix, cut it
        input_ids = input_ids[:, prefix_length:]
        return input_ids.tolist()
    
    @torch.no_grad()
    def predict_contrastive_search(self, batch):
        self.model.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        for step in range(self.test_max_len):
            input_ids = ContrastiveDecodingOneStep(
                self.model,
                input_ids,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.args['sampling_probability'],
                self.sep,
                min(1., (step+1)/self.args['sep_smooth_length']),
            )
        # input_ids contains the prefix, cut it
        input_ids = input_ids[:, prefix_length:]
        return input_ids.tolist()

    @torch.no_grad()
    def predict_beam_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids, 
            prefix_length+self.test_max_len, 
            num_beams=self.args['num_beam'],
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
        )
        beam_output = beam_output[:, prefix_length:]
        return beam_output.tolist()
    
    @torch.no_grad()
    def predict_greedy_search(self, batch):
        '''batch_size is 1'''
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
            )[0]    # [1, S, V]
            next_token_logits = output[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(0)
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            ids = ids[:, -self.test_max_ctx_len:]
        return generated

    @torch.no_grad()
    def predict_topk_topp_repetition_penalty_batch_fast(self, batch):
        '''topk-topp search with batch inference, pad in the left'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.model(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(
                next_token_logits,
                top_k=self.topk,
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1
            )
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        return generated
    
    @torch.no_grad()
    def predict_topk_topp_repetition_penalty_fast(self, batch):
        ids = batch['ids']
        generated = []
        past_key_values = None
        while True:
            output = self.model(
                input_ids=ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = next_token.unsqueeze(0)
        return [generated]

    @torch.no_grad()
    def predict_topk_topp_repetition_penalty(self, batch):
        '''batch_size is 1'''
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
            )[0]    # [1, S, V]
            next_token_logits = output[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
        return [generated]
    
    @torch.no_grad()
    def predict_topk_topp(self, batch):
        '''batch_size is 1'''
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids, 
            prefix_length+self.test_max_len, 
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            top_p=self.topp,
            top_k=self.topk,
            do_sample=True,
        )
        beam_output = beam_output[:, prefix_length:]
        return beam_output.tolist()
    
    @torch.no_grad()
    def predict_topp(self, batch):
        '''batch_size is 1'''
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids, 
            prefix_length+self.test_max_len, 
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            top_p=self.topp,
            top_k=0,
            do_sample=True,
        )
        beam_output = beam_output[:, prefix_length:]
        return beam_output.tolist()
    
    @torch.no_grad()
    def predict_topk(self, batch):
        '''batch_size is 1'''
        ids = batch['ids']
        _, prefix_length = ids.size()
        beam_output = self.model.generate(
            ids, 
            prefix_length+self.test_max_len, 
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            top_k=self.topk,
            do_sample=True,
        )
        beam_output = beam_output[:, prefix_length:]
        return beam_output.tolist()
