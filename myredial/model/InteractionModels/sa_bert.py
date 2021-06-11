from model.utils import *

'''Speraker-aware Bert Cross-encoder'''

class SABERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(SABERTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.head = nn.Linear(768, 2)
        self.speaker_embedding = nn.Embedding(2, 768)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        speaker_type_ids = batch['sids']
        attn_mask = batch['mask']

        word_embeddings = self.model.embeddings(inpt)    # [B, S, 768]
        speaker_embedding = self.speaker_embedding(speaker_type_ids)   # [B, S, 768]
        word_embeddings += speaker_embedding
        output = self.model(
            input_ids=None,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=word_embeddings,
        )[0]    # [B, S, E]
        logits = self.head(output[:, 0, :])    # [B, H] -> [B, 2]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)
