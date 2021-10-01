from model.utils import *
from dataloader.util_func import *

class BERTCompareRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareRetrieval, self).__init__()
        model = args['pretrained_model']
        self.inner_bsz = args['inner_bsz']
        self.num_labels = args['num_labels']
        self.model = SABertForSequenceClassification.from_pretrained(model, num_labels=self.num_labels)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

    def forward(self, batch, scaler=None, optimizer=None, scheduler=None, grad_clip=1.0):
        inpt = batch['ids']
        sids = batch['sids']
        tids = batch['tids']
        label = batch['label']    # list

        # shuffle
        random_idx = list(range(len(inpt)))
        random.shuffle(random_idx)
        inpt = [inpt[i] for i in random_idx]
        sids = [sids[i] for i in random_idx]
        tids = [tids[i] for i in random_idx]
        label = [label[i] for i in random_idx]
        label = torch.stack(label)

        acc, tloss, counter = 0, 0, 0
        for i in range(0, len(inpt), self.inner_bsz):
            sub_ids = pad_sequence(
                inpt[i:i+self.inner_bsz],
                batch_first=True,
                padding_value=self.pad,
            )
            sub_sids = pad_sequence(
                sids[i:i+self.inner_bsz],
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
            sub_label = sub_label.to(torch.float)

            sub_ids, sub_sids, sub_tids, sub_attn_mask, sub_label = to_cuda(sub_ids, sub_sids, sub_tids, sub_attn_mask, sub_label)
            with autocast():
                logits = self.model(
                    input_ids=sub_ids,
                    attention_mask=sub_attn_mask,
                    token_type_ids=sub_tids,
                    speaker_ids=sub_sids,
                )[0]
                logits = logits.squeeze()     # [B]
                loss = self.criterion(logits, sub_label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tloss += loss
            acc += torch.sum((torch.sigmoid(logits) > 0.5) == sub_label).item()/len(sub_label)
            counter += 1
        tloss /= counter
        return tloss, acc, counter

    def predict(self, batch):
        inpt = batch['ids']
        sids = batch['sids']
        tids = batch['tids']
        mask = batch['mask']   
        logits = self.model(
            input_ids=inpt,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]
        logits = torch.sigmoid(logits.squeeze(dim=-1))    # [B]
        return logits
