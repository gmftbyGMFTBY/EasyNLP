from .header import *
from inference_utils import *

class GPT2CLHeadModel(nn.Module):

    def __init__(self, model_name, vocab_size, 
        unk=None, pad=None, cls=None, sep=None, temp=0.07
    ):
        super(GPT2CLHeadModel, self).__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.lm = nn.Parameter(torch.randn(768, vocab_size))
        self.vocab_size = vocab_size
        self.unk = unk
        self.pad = pad
        self.cls = cls
        self.sep = sep
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)

    def forward(self, ids, ids_mask):
        '''ids/ids_mask: [B, S]; label: [B, S]'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
        )
        rep = F.normalize(output.last_hidden_state, dim=-1)    # [B, S, E]
        rep = rep[:, :-1, :].contiguous()    # [B, S-1, E]
        ids_mask = ids_mask[:, :-1]    # [B, S-1]
        label = ids[:, 1:]    # [B, S-1]

        # contrastive loss
        dp = torch.matmul(rep, self.lm).view(-1, self.vocab_size)    # [B*S, V]
        dp /= self.temp
        mask = torch.zeros_like(dp)
        mask[range(len(dp)), label.reshape(-1)] = 1.    # [B*S, V]
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = -loss_.sum(dim=1)     # [B*S]
        # only non-padding tokens will be used for training
        loss = loss[~(ids_mask.reshape(-1) == self.pad)].mean()
        
        # acc (ignore padding tokens)
        acc = dp.max(dim=-1)[1] == label.reshape(-1)
        acc = acc[~(ids_mask.reshape(-1) == self.pad)].to(torch.float).mean().item()
        return loss, acc

    def predict_one_step(self, ids, ids_mask, pos_ids, past_key_values):
        '''use cache for speedup'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        rep = F.normalize(output.last_hidden_state, dim=-1)    # [B, S, E]
        past_key_values = output.past_key_values

        scores = torch.matmul(rep, self.lm)    # [B, S, V]
        next_token_logits = scores[:, -1, :]
        # special tokens not used
        next_token_logits[:, self.unk] = -np.inf
        next_token_logits[:, self.pad] = -np.inf
        next_token_logits[:, self.sep] = -np.inf
        next_token_logits[:, self.cls] = -np.inf

        next_token = next_token_logits.max(dim=-1)[1]    # [B]
        return next_token, past_key_values

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        rep = self.model(
            input_ids=ids, 
            attention_mask=ids_mask
        ).last_hidden_state
        gen_logits = torch.matmul(rep, self.lm)    # [B, S, V]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl


class GPT2CLFromLMHeadModel(nn.Module):

    def __init__(self, model_name, vocab_size, 
        unk=None, pad=None, cls=None, sep=None, temp=0.07
    ):
        super(GPT2CLFromLMHeadModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.lm = nn.Parameter(
            self.model.lm_head.state_dict()['weight'].t()
        )
        self.vocab_size = vocab_size
        self.unk = unk
        self.pad = pad
        self.cls = cls
        self.sep = sep
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)

    def forward(self, ids, ids_mask):
        '''ids/ids_mask: [B, S]; label: [B, S]'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            output_hidden_states=True,
        )
        rep = F.normalize(output.hidden_states[-1], dim=-1)    # [B, S, E]
        rep = rep[:, :-1, :].contiguous()    # [B, S-1, E]
        ids_mask = ids_mask[:, :-1]    # [B, S-1]
        label = ids[:, 1:]    # [B, S-1]

        # contrastive loss
        dp = torch.matmul(rep, self.lm).view(-1, self.vocab_size)    # [B*S, V]
        dp /= self.temp
        mask = torch.zeros_like(dp)
        mask[range(len(dp)), label.reshape(-1)] = 1.    # [B*S, V]
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss = -loss_.sum(dim=1)     # [B*S]
        # only non-padding tokens will be used for training
        loss = loss[~(ids_mask.reshape(-1) == self.pad)].mean()
        
        # acc (ignore padding tokens)
        acc = dp.max(dim=-1)[1] == label.reshape(-1)
        acc = acc[~(ids_mask.reshape(-1) == self.pad)].to(torch.float).mean().item()
        return loss, acc

    def predict_one_step(self, ids, ids_mask, pos_ids, past_key_values):
        '''use cache for speedup'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        rep = F.normalize(output.hidden_states[-1], dim=-1)    # [B, S, E]
        past_key_values = output.past_key_values

        scores = torch.matmul(rep, self.lm)    # [B, S, V]
        next_token_logits = scores[:, -1, :]
        # special tokens not used
        next_token_logits[:, self.unk] = -np.inf
        next_token_logits[:, self.pad] = -np.inf
        next_token_logits[:, self.sep] = -np.inf
        next_token_logits[:, self.cls] = -np.inf
        next_token = next_token_logits.max(dim=-1)[1]    # [B]
        return next_token, past_key_values
    
    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        rep = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        gen_logits = torch.matmul(rep, self.lm)    # [B, S, V]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl


class GPT2CLSDModel(nn.Module):

    '''static embeddings and dynamic embeddings'''

    def __init__(self, model_name, bert_model_name, 
        unk=None, pad=None, cls=None, sep=None, temp=0.07,
        dropout=0.1, index_type='Flat', dimension=768, nprobe=1,
        faiss_path=None, corpus_path=None, max_phrase_len=10,
        coarse_selection_topk=10, length_penalty=1.1,
    ):
        super(GPT2CLSDModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # character static embeddings
        self.vocab = AutoTokenizer.from_pretrained(model_name)
        self.lm = nn.Parameter(
            self.model.lm_head.state_dict()['weight'].t()
        )    # [E, V]
        self.vocab_size = len(self.vocab)
        self.unk = unk
        self.pad = pad
        self.cls = cls
        self.sep = sep
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        # dynamic static embeddings
        self.retrieval = BertModel.from_pretrained(bert_model_name)
        self.phrase_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # essential parameters
        self.length_penalty = length_penalty
        self.coarse_selection_topk = coarse_selection_topk
        self.max_phrase_len = max_phrase_len

        # faiss index
        self.cached_index = PhraseSearcher(
            index_type=index_type,
            dimension=dimension,
            nprobe=nprobe
        )
        self.faiss_path, self.corpus_path = faiss_path, corpus_path

    @torch.no_grad()
    def offline_inference(self, batch):
        '''generate the offline represetations'''
        self.retrieal.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
        text, index_in_ids, index_in_text = batch['text'], batch['einid'], batch['eintext']
        rep = self.retrieval(ids, ids_mask).last_hidden_state    # [B, S, E]
        # rest:
        embd, rest_text, rest_eintext = [], [], []
        for rep_, einid, eintext, t in zip(rep, index_in_id, index_in_text, text):
            # rep_: [S, E]
            embd.append(rep_[einid, :])
            rest_text.extend([t] * len(einid))
            rest_eintext.extend(eintext)
        embd = torch.cat(embd)
        assert len(embd) == len(rest_text) == len(rest_eintext)
        return embd, rest_text, rest_eintext

    def get_ppl_for_each_step(self, ids, ids_mask, label):
        '''ids/ids_mask: [B, S], only the candidates will be used to calculate the loss (ppl), prefix and padding are set as -100, which will be ignored
        return the probablity of each step in each sentence
        '''
        rep = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True
        ).hidden_states[-1]    # [B, S, E]
        bsz, length, _ = rep.size()
        logits = torch.matmul(rep, self.lm)    # [B, S, V]
        shift_logits = logits[..., :-1, :].contiguous()    # [B, S-1, V]
        shift_labels = label[..., 1:].contiguous()    # [B, S-1]
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).reshape(bsz, length-1)    # [B, S-1]
        # moving average with length penalty
        ppls = []
        for i in range(bsz):
            loss_ = loss[i].nonzero().squeeze(-1).exp()    # [S']
            weight = torch.tensor([self.length_penalty] * len(loss_)).cumprod(dim=-1)
            loss_ = (loss_ * weight).cumsum(dim=-1)    # [S']
            div_weight = torch.arange(1, 1 + len(loss_))
            loss_ = loss_ / div_weight
            ppls.append(loss_)
        return ppls

    def forward(self, ids, ids_mask):
        '''ids/ids_mask: [B, S]; label: [B, S]'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            output_hidden_states=True,
        )
        rep = output.hidden_states[-1]
        ## token contrastive loss (cross-entropy-based)
        # this objective serves two purpose: 
        # (1) fine-grained rerank for phrase candidate
        # (2) token-level contrastive loss or causal language model
        gen_logits = torch.matmul(rep, self.lm)    # [B, S, V]
        shift_logits = gen_logits[..., :-1, :].contiguous() 
        shift_labels = ids[..., 1:].contiguous()
        loss_token = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # token acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        token_acc = correct / num_targets

        ## phrase cross-entropy loss (logits: [B, S-1, S-1])
        dy_rep = self.retrieval(
            input_ids=ids,
            attention_mask=ids_mask,
        ).last_hidden_state
        dy_rep = dy_rep[..., 1:, :]    # [B, S-1, E]
        query_rep = rep[..., :-1, :]   # [B, S-1, E]
        gen_logits = torch.bmm(query_rep, dy_rep.permute(0, 2, 1))    # [B, S-1, S-1]
        bsz, length, _ = gen_logits.size()
        gen_labels = torch.arange(length).cuda().unsqueeze(0).expand(bsz, -1)    # [B, S-1]
        gen_labels = gen_labels.masked_fill(~ids_mask[:, :-1].to(torch.bool), -100)
        # padding tokens must be ignored
        loss_phrase = self.phrase_criterion(
            gen_logits.view(-1, gen_logits.size(-1)),    # [B*(S-1), S-1]
            gen_labels.view(-1),    # B*(S-1)
        )
        # phrase acc
        _, preds = gen_logits.max(dim=-1)    # [B, S-1]
        not_ignore = gen_labels != -100    # [B, S-1]
        num_targets = not_ignore.long().sum().item()
        correct = (gen_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        phrase_acc = correct / num_targets
        return loss_token, loss_phrase, token_acc, phrase_acc

    def predict_one_step_retrieval_version(self, ids, ids_mask, pos_ids, past_key_values):
        '''two step pipeline retrieval:
            1. coarse-grained retrieval: (a) token-level retrieval; (b) phrase-level retrieval
            2. fine-grained rerank: language model selection'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        rep = output.hidden_states[-1][:, -1, :]    # [B, E]
        past_key_values = output.past_key_values

        ## 1. coarse-grained retrieval
        #  1.1 token-level
        token_logits = torch.matmul(rep, self.lm)
        topk_prob, topk_ids = token_logits.topk(self.coarse_selection_topk, dim=-1)    # [B, K]
        #  1.2 phrase-level recall
        # rest: [B, K]
        rest = self.cached_index._search(
            rep.numpy(), 
            topk=self.coarse_selection_topk, 
            max_phrase_len=self.max_phrase_len
        )
        ipdb.set_trace()

        ## 1.x build the batch for fine-grained rerank

        ## 2. fine-grained rerank
        return next_token, past_key_values

    def predict_one_step(self, ids, ids_mask, pos_ids, past_key_values):
        '''use cache for speedup'''
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        rep = output.hidden_states[-1]    # [B, S, E]
        past_key_values = output.past_key_values

        scores = torch.matmul(rep, self.lm)    # [B, S, V]
        next_token_logits = scores[:, -1, :]
        # special tokens not used
        next_token_logits[:, [self.pad, self.sep, self.cls, self.unk]] = -np.inf
        next_token = next_token_logits.max(dim=-1)[1]    # [B]
        return next_token, past_key_values
    
    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        rep = self.model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        gen_logits = torch.matmul(rep, self.lm)    # [B, S, V]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl
