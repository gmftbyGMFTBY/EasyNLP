from model.utils import *

class RAGBERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(RAGBERTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)

        sel.head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(768*2, 2)
        )

    def get_rag_rep(self, batch):
        rag_rep = self.model(input_ids=batch['rag_ids'], token_type_ids=batch['rag_tids'], attention_mask=batch['rag_mask'])[:, 0, :]     # [B*K, E]
        rag_rep = torch.stackc(torch.split(rag_rep, batch['split_size'], dim=0))    # [B, K, E]
        rag_rep = rag_rep.mean(dim=1)   # [B, E]
        return rag_rep
        
    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        rag_rep = self.get_rag_rep(batch)    # [B, E]

        rep = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0, :]    # [B, E]

        hidden_state = torch.cat([rep, rag_rep], dim=-1)    # [B, 2*E]
        logits = self.head(hidden_state)    # [B, 2]
        return logits
