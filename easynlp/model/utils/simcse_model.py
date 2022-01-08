import torch.nn as nn
from transformers import BertConfig, BertModel

class simcse_model(nn.Module):

    def __init__(self, 
        pretrained_model="hfl/chinese-bert-wwm-ext", 
        dropout=0.1
    ):
        super(simcse_model, self).__init__()
        conf = BertConfig.from_pretrained(pretrained_model)
        conf.attention_probs_dropout_prob = dropout
        conf.hidden_dropout_prob = dropout
        self.encoder = BertModel.from_pretrained(pretrained_model, config=conf)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        return output.last_hidden_state
