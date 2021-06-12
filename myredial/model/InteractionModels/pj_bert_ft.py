from model.utils import *

class PJBERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(PJBERTRetrieval, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']

        self.model = PJBertModel.from_pretrained(model, **args)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        logits = self.model(inpt, token_type_ids)    # [B]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)
