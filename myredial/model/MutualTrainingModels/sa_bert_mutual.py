from model.utils import *

class SABERTMutualRetrieval(nn.Module):

    def __init__(self, **args):
        super(SABERTMutualRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertSAModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.cls = nn.Linear(768, 1)

    def forward(self, batch):
        '''soft label training needs the BCELoss'''
        inpt = batch['ids']
        token_type_ids = batch['tids']
        speaker_type_ids = batch['sids']
        attn_mask = batch['mask']
        soft_label = batch['soft_label']

        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            speaker_ids=speaker_type_ids,
        )[0]    # [B, S, E]
        logits = self.cls(logits[:, 0, :]).squeeze(dim=-1)    # [B]
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, soft_label)
        return loss

    @torch.no_grad()
    def predict(self, batch):
        ids, tids, sids, mask = batch['ids'], batch['tids'], batch['sids'], batch['mask']
        logits = self.model(input_ids=ids, attention_mask=mask, token_type_ids=tids, speaker_ids=sids)[0]
        logits = torch.sigmoid(self.cls(logits[:, 0, :])).squeeze(dim=-1)    # [B]
        return logits
