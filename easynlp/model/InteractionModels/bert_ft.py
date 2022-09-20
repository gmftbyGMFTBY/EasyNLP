from model.utils import *

class BERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTRetrieval, self).__init__()
        model = args['pretrained_model']
        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
        # self.model = ElectraForSequenceClassification.from_pretrained(model, num_labels=2)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.vocab = AutoTokenizer.from_pretrained(model)

        total = sum([param.nelement() for param in self.parameters()])
        print('[!] Model Size: %2fM' % (total/1e6))

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, 2]

        return logits

    @torch.no_grad()
    def predict(self, batch):
        hidden = self.model(input_ids=batch['ids'], token_type_ids=batch['tids'], attention_mask=batch['mask'], output_hidden_states=True)['hidden_states'][-1][:, 0, :]     # [B, E]
        return hidden
