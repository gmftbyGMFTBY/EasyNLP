from model.utils import *

class BERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTRetrieval, self).__init__()
        model = args['pretrained_model']
        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, 1]
        return logits.squeeze(dim=-1)

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        old_state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            if k.startswith('model.'):
                k = k.replace('model.', '')
                if k.startswith('cls.'):
                    continue
            else:
                # encoder.layer.0 -> bert.encoder.layer.0; ...
                k = f'bert.{k}'
            new_state_dict[k] = v
        old_state_dict.update(new_state_dict)
        self.model.load_state_dict(old_state_dict)
