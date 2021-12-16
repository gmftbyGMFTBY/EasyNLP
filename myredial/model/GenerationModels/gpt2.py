from model.utils import *
from .utils import *


class InferenceGPT2Model(nn.Module):

    '''only for inference testing, not trained'''

    def __init__(self, **args):
        super(InferenceGPT2Model, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        # pad token is 0
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.test_max_len = args['gen_max_len']
        self.test_max_ctx_len = args['gen_max_ctx_len']
        self.repetition_penalty = args['repetition_penalty']
        if args['decoding_method'] == 'contrastive_search':
            self.predict = self.predict_contrastive_search
        elif args['decoding_method'] == 'greedy_search':
            self.predict = self.predict_greedy_search
        elif args['decoding_method'] == 'beam_search':
            self.predict = self.predict_beam_search
        elif args['decoding_method'] == 'topk_topp_repetition_penalty_search':
            self.predict = self.predict_topk_topp_repetition_penalty
        elif args['decoding_method'] == 'topk_topp_search':
            self.predict = self.predict_topk_topp
        else:
            raise Exception(f'[!] cannot find the deocidng method: {args["decoding_method"]}')
        self.args = args

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
    def predict_contrastive_search(self, batch):
        self.model.eval()
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        for step in range(self.test_max_len):
            input_ids = ContrastiveDecodingOneStep(
                self.model,
                input_ids,
                self.args['beam_width'],
                self.args['scoring_criterion'],
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
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
        return [generated]
    
    @torch.no_grad()
    def predict_topk_topp(self, batch):
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
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
        return [generated]
