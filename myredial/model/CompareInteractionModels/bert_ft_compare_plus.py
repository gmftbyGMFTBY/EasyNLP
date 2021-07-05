from model.utils import *

'''BCE Loss to three classification'''

class BERTComparePlusRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTComparePlusRetrieval, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']
        self.inner_bsz = args['inner_bsz']
        self.model = BertModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

        hidden_size = self.model.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(hidden_size, 3)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

    def forward(self, batch, scaler=None, optimizer=None):
        inpt = batch['ids']
        tids = batch['tids']
        label = batch['label']    # list

        # shuffle
        random_idx = list(range(len(inpt)))
        random.shuffle(random_idx)
        inpt = [inpt[i] for i in random_idx]
        tids = [tids[i] for i in random_idx]
        label = [label[i] for i in random_idx]
        label = torch.stack(label)

        tloss = 0
        outputs = []
        for i in range(0, len(inpt), self.inner_bsz):
            sub_ids = pad_sequence(
                inpt[i:i+self.inner_bsz],
                batch_first=True,
                padding_value=self.pad,
            )
            sub_tids = pad_sequence(
                tids[i:i+self.inner_bsz],
                batch_first=True,
                padding_value=self.pad,
            )
            sub_attn_mask = self.generate_mask(sub_ids)
            sub_label = label[i:i+self.inner_bsz]

            if torch.cuda.is_available():
                sub_ids = sub_ids.cuda()
                sub_tids = sub_tids.cuda()
                sub_attn_mask = sub_attn_mask.cuda()
                sub_label = sub_label.cuda()

            output = self.model(
                input_ids=sub_ids,
                attention_mask=sub_attn_mask,
                token_type_ids=sub_tids,
            )[0]    # [B, S, E]
            logits = self.head(output[:, 0, :])    # [B, 3]
            outputs.append(logits)
            loss = self.criterion(logits, sub_label)
            tloss += loss
            # backward and gradient accumulation
            scaler.scale(loss).backward()
        output = torch.cat(outputs)    # [B, 3]
        return tloss, output

    def predict(self, batch):
        inpt = batch['ids']
        tids = batch['tids']
        mask = batch['mask']   
        output = self.model(
            input_ids=sub_ids,
            attention_mask=sub_attn_mask,
            token_type_ids=sub_tids,
        )[0]    # [B, S, E]
        logits = F.softmax(self.head(output[:, 0, :]), dim=-1)    # [B, 3]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)

    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
