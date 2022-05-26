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
            self.unk = self.vocab.unk_token_id
            self.special_tokens = set([self.pad])
        else:
            self.unk, self.pad, self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
            self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.test_max_len = args['test_max_len']

        if self.args['mode'] == 'test':
            # load the faiss index
            pass

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
