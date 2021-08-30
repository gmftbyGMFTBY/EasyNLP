from model.utils import *

'''Speraker-aware Bert Cross-encoder'''

class SABERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(SABERTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.speaker_embedding = nn.Embedding(2, 768)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        speaker_type_ids = batch['sids']
        attn_mask = batch['mask']

        word_embeddings = self.model.bert.embeddings(inpt)    # [B, S, 768]
        speaker_embedding = self.speaker_embedding(speaker_type_ids)   # [B, S, 768]
        word_embeddings += speaker_embedding
        logits = self.model(
            input_ids=None,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=word_embeddings,
        )[0]    # [B, 1]
        return logits.squeeze(dim=-1)
