from model.utils import *

class BERTIBNSRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTIBNSRetrieval, self).__init__()
        model = args['pretrained_model']
        self.gray_cand_num = args['gray_cand_num'] + 1
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        score = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0].squeeze(dim=-1)    # [B]

        return score
