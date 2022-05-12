from model.utils import *

class BERTMutualDatasetRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTMutualDatasetRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)
        self.vocab = AutoTokenizer.from_pretrained(model)

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
