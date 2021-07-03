from model.utils import *

class BERTCompareRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.head = nn.Linear(768, 1)
        # 0: context; 1: candidate1; 2: candidate2
        self.seg_embedding = nn.Embedding(3, 768)

        # for debug
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')

    def forward(self, batch):
        inpt = batch['ids']
        seg_ids = batch['tids']
        attn_mask = batch['mask']

        word_embeddings = self.model.embeddings(inpt)    # [B, S, 768]
        seg_embedding = self.seg_embedding(seg_ids)   # [B, S, 768]
        word_embeddings += seg_embedding
        output = self.model(
            input_ids=None,
            attention_mask=attn_mask,
            inputs_embeds=word_embeddings,
        )[0]    # [B, S, E]
        logits = self.head(output[:, 0, :]).squeeze(-1)    # [B]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)
