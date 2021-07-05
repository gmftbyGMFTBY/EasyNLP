from model.utils import *

class BERTCompareRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareRetrieval, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 1)
        )

    def forward(self, batch):
        inpt = batch['ids']
        seg_ids = batch['tids']
        attn_mask = batch['mask']

        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=seg_ids,
        )[0]    # [B, S, E]
        logits = self.head(output[:, 0, :]).squeeze(-1)    # [B]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)
