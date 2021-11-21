from model.utils import *

class BERTSCMRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTSCMRetrieval, self).__init__()
        model = args['pretrained_model']
        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertModel.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 1)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args['num_layers'])
        self.criterion = nn.BCEWithLogitsLoss()
        self.topk = args['gray_cand_num_hn'] + args['gray_cand_num_en'] + 1

    def forward(self, batch):
        inpt = batch['ids']    # [B*K, S]
        token_type_ids = batch['tids']    # [B*K, S]
        attn_mask = batch['mask']    # [B*K, S]

        embds = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0, :]    # [B*K, E]
        embds = torch.stack(torch.split(embds, self.topk))    # [B, K, E]
        embds_ = self.fusion_encoder(
            embds.permute(1, 0, 2),    # [K, B, E]
        ).permute(1, 0, 2)    # [B, K, E]
        embds = embds + embds_
        opt = self.cls_head(embds).squeeze(-1)    # [B, K]
        label = torch.zeros_like(opt)
        label[:, 0] = 1.
        loss = self.criterion(opt.view(-1), label.view(-1))

        acc = ((torch.sigmoid(opt).view(-1) > 0.5) == label.view(-1)).to(torch.float).mean().item()
        return loss, acc

    def predict(self, batch):
        inpt = batch['ids']    # [K, S]
        token_type_ids = batch['tids']    # [K, S]
        attn_mask = batch['mask']    # [K, S]

        embds = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0, :]    # [K, E]
        embds = embds.unsqueeze(0)    # [1, K, E]
        embds = self.fusion_encoder(
            embds.permute(1, 0, 2),    # [K, 1, E]
        ).permute(1, 0, 2)    # [1, K, E]
        opt = self.cls_head(embds).squeeze(0).squeeze(-1)    # [K]
        opt = torch.sigmoid(opt)
        return opt
