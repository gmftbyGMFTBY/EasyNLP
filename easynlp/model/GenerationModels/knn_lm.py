from model.utils import *
from .utils import *


class KNNLMModel(nn.Module):

    '''GPT-2 based KNN-LM model'''

    def __init__(self, **args):
        super(KNNLMModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.vocab)
        self.args = args

        if args['lang'] == 'en':
            self.pad = self.vocab.eos_token_id
            self.unk = self.vocab.eos_token_id
            self.special_tokens = set([self.pad])
        else:
            self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
            self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.test_max_len = args['test_max_len']

    def init_searcher(self, searcher):
        self.seacher = searcher

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask):
        ids, ids_mask, label = ids[:, :-1], ids_mask[:, :-1], ids[1:, :]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        logits = output.logits
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
        return math.exp(loss.item())

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        vl = ids_mask.sum(dim=-1)
        collection_rep, collection_target = [], []
        ids = ids.tolist()
        for rep, ids_, l in zip(output, ids, vl):
            # rep: [S, E]
            collection_rep.append(rep[:l-1, :])
            collection_target.extend(ids_[1:l])
        collection_rep = torch.cat(collection_rep).cpu()
        assert len(collection_rep) == len(collection_target)
        collection_target = [str(i) for i in collection_target]
        return collection_rep, collection_target

    @torch.no_grad()
    def generate_new_logits(self, logits, hidden, topk=10):
        cands, dists = self.searcher.search_dis(hidden.numpy(), topk=topk)
        tokens = self.vocab.convert_tokens_to_ids(cands)
        knn_logits = torch.zeros(len(self.vocab)).cuda().unsqueeze(0).expand(topk, -1)    # [K, V]
        knn_logits[range(topk), tokens] = torch.exp(-torch.from_numpy(dists))
        knn_logits = F.softmax(knn_logits.sum(dim=0), dim=-1)    # [V]
        ipdb.set_trace()
        new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * logits
        return new_logits

    @torch.no_grad()
    def greedy_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden = output['hidden_states'][-1][-1, -1, :]    # [H]
            next_token_logits = output['logits'][-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf

            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['topk'])

            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(0)
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            ids = ids[:, -self.test_max_ctx_len:]
        string = self.vocab.decode(generated)
        return string

    @torch.no_grad()
    def topk_topp_search(self, batch):
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
                output_hidden_states=True
            )
            hidden = output['hidden_states'][-1][-1, -1, :]    # [H]
            next_token_logits = output['logits'][-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['topk'])
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
            ids = ids[:, -self.test_max_ctx_len:]
        string = self.vocab.decode(generated)
        return string
