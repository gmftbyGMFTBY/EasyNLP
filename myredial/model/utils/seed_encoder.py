from .header import *


class WeakTrsDecoder(nn.Module):
    
    def __init__(self, dropout, vocab_size, nhead, nlayer, attention_span):
        super(WeakTrsDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayer)
        self.lm_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(768, vocab_size)
        )
        self.attn_span_k = attention_span
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self.attn_span_mask = torch.ones(512, 512)
        for i in range(512):
            index = i - self.attn_span_k if i - self.attn_span_k > 0 else 0
            self.attn_span_mask[i, :index] = 0.
        self.attn_span_mask = self.attn_span_mask.to(torch.bool)

    def forward(self, ids, memory):
        # ids: [B, S]
        embedding = self.embedding(ids).permute(1, 0, 2)    # [S, B, E]
        # memory: [1, B, E]; only the [CLS] token is used
        weak_mask = self.generate_weak_attn_mask(ids.size(1))
        outputs = self.decoder(
            tgt=embedding,
            tgt_mask=weak_mask,
            memory=memory,
        )    # [S, B, E]
        outputs = self.lm_head(outputs).permute(1, 0, 2)    # [B, S, V]
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)) ,
            shift_labels.view(-1),
        )
        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, acc

    def generate_weak_attn_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask & self.attn_span_mask[:length, :length]
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask


class BertSEEDEmbedding(nn.Module):
    
    def __init__(
        self, 
        model='bert-base-chinese', 
        add_tokens=1, 
        dropout=0.1,
        vocab_size=0,
        nhead=12,
        nlayer=3,
        attn_span=3
    ):
        super(BertSEEDEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.decoder = WeakTrsDecoder(
            dropout,
            vocab_size,
            nhead,
            nlayer,
            attn_span,
        )
        self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask):
        embds = self.model(ids, attention_mask=attn_mask, output_hidden_states=True)[0]    # [B, S, E]
        loss, acc = self.decoder(ids, embds.permute(1, 0, 2)[:1, :, :], attn_mask)
        return embds[:, 0, :], loss, acc
