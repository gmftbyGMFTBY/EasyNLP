from model.utils import *

'''The original BERT-FP Model and the checkpoint (do not train it)'''

class BERTFPRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTFPRetrieval, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']

        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    @torch.no_grad()
    def forward(self, batch):
        '''only for inferencing and testing'''
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, 1]
        output = torch.sigmoid(output).squeeze()    # [B]
        return output

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace('bert_model.', '')
            new_state_dict[k] = v
        # compatible with the old bert-fp pre-trained checkpoint
        new_state_dict['bert.embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)
