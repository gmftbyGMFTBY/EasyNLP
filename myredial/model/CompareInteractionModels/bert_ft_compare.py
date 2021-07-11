from model.utils import *
from dataloader.util_func import *

class BERTComparePlusRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTComparePlusRetrieval, self).__init__()
        model = args['pretrained_model']
        self.inner_bsz = args['inner_bsz']
        self.num_labels = args['num_labels']
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=self.num_labels)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        if self.num_labels == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.vocab.add_tokens(['[EOS]'])
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
        avg = len(inpt) // self.inner_bsz
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
            sub_attn_mask = generate_mask(sub_ids)
            sub_label = label[i:i+self.inner_bsz]

            sub_ids, sub_tids, sub_attn_mask, sub_label = to_cuda(sub_ids, sub_tids, sub_attn_mask, sub_label)
            with autocast():
                logits = self.model(
                    input_ids=sub_ids,
                    attention_mask=sub_attn_mask,
                    token_type_ids=sub_tids,
                )[0]    # [B, 3] or [B, 1]
                if self.num_labels == 1:
                    logits = logits.squeeze()     # [B]
                    sub_label = sub_label.to(torch.float)
                loss = self.criterion(logits, sub_label)
                loss /= avg
            # backward and gradient accumulation
            scaler.scale(loss).backward()
            tloss += loss
            outputs.append(logits)
        output = torch.cat(outputs)    # [B, 3]; [B]
        return tloss, output, label

    def predict(self, batch):
        inpt = batch['ids']
        tids = batch['tids']
        mask = batch['mask']   
        logits = self.model(
            input_ids=inpt,
            attention_mask=mask,
            token_type_ids=tids,
        )[0]    # [B, 3] / [B, 1]
        if self.num_labels == 1:
            logits = torch.sigmoid(logits.squeeze(dim=-1))    # [B]
        else:
            logits = F.softmax(logits, dim=-1)    # [B, 3]
        return logits

    def load_bert_model(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.bert.load_state_dict(new_state_dict)
