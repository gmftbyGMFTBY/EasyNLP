from model.utils import *
from dataloader.util_func import *


class BERTCompareRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareRetrieval, self).__init__()
        model = args['pretrained_model']
        self.inner_bsz = args['inner_bsz']
        self.num_labels = args['num_labels']
        self.model = SABertForSequenceClassification.from_pretrained(model, num_labels=2)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss()

        # vocabulary
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

    def forward(self, batch, scaler=None, optimizer=None, scheduler=None, grad_clip=1.0):
        inpt = batch['ids']
        tids = batch['tids']
        cpids = batch['cpids']
        label = batch['label']    # list

        # shuffle
        random_idx = list(range(len(inpt)))
        random.shuffle(random_idx)
        inpt = [inpt[i] for i in random_idx]
        tids = [tids[i] for i in random_idx]
        cpids = [cpids[i] for i in random_idx]
        label = [label[i] for i in random_idx]
        label = torch.stack(label)

        token_acc, acc, tloss, counter = 0, 0, 0, 0
        for i in range(0, len(inpt), self.inner_bsz):
            optimizer.zero_grad()
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
            sub_cpids = pad_sequence(
                cpids[i:i+self.inner_bsz],
                batch_first=True,
                padding_value=self.pad,
            )
            sub_attn_mask = generate_mask(sub_ids)
            sub_label = label[i:i+self.inner_bsz]

            sub_ids, sub_cpids, sub_tids, sub_attn_mask, sub_label = to_cuda(sub_ids, sub_cpids, sub_tids, sub_attn_mask, sub_label)
            with autocast():
                output = self.model(
                    input_ids=sub_ids,
                    attention_mask=sub_attn_mask,
                    token_type_ids=sub_tids,
                    compare_ids=sub_cpids
                )
                logits = F.sigmoid(output.logits).squeeze(dim=-1)    # [B]
                loss = self.criterion(logits, sub_label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tloss += loss
            acc += ((logits > 0.5) == (sub_label > 0.5)).to(torch.float).mean().item()
            counter += 1
        tloss /= counter
        acc /= counter
        return tloss, acc

    def predict(self, batch):
        inpt = batch['ids']
        tids = batch['tids']
        cpids = batch['cpids']
        mask = batch['mask']   
        output = self.model(
            input_ids=inpt,
            attention_mask=mask,
            token_type_ids=tids,
            compare_ids=cpids,
        )
        logit = F.sigmoid(output.logits).squeeze(dim=-1)
        return logit


class BERTCompareTokenEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTCompareTokenEncoder, self).__init__()
        model = args['pretrained_model']
        self.inner_bsz = args['inner_bsz']
        self.model = BertSAModel.from_pretrained(model)
        self.cls = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 2)
        )
        # add the [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss()

        # vocabulary
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

    def forward(self, batch, scaler=None, optimizer=None, scheduler=None, grad_clip=1.0):
        ids = batch['ids']
        sids = batch['sids']
        tids = batch['tids']
        tlids = batch['tlids']

        # shuffle
        random_idx = list(range(len(ids)))
        random.shuffle(random_idx)
        ids  = [ids[i] for i in random_idx]
        sids = [sids[i] for i in random_idx]
        tids = [tids[i] for i in random_idx]
        tlids = [tlids[i] for i in random_idx]

        token_acc, acc, tloss, counter = 0, 0, 0, 0
        for i in range(0, len(ids), self.inner_bsz):
            sub_ids = pad_sequence(
                ids[i:i+self.inner_bsz],
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
            sub_tlids = pad_sequence(
                tlids[i:i+self.inner_bsz],
                batch_first=True,
                padding_value=-100,
            )
            sub_attn_mask = generate_mask(sub_ids)

            sub_ids, sub_sids, sub_tids, sub_attn_mask, sub_tlids = to_cuda(sub_ids, sub_sids, sub_tids, sub_attn_mask, sub_tlids)
            with autocast():
                output = self.model(
                    input_ids=sub_ids,
                    attention_mask=sub_attn_mask,
                    token_type_ids=sub_tids,
                    speaker_ids=sub_sids,
                )[0]
                logits = self.cls(output)     # [B, S, 2]
                loss = self.criterion(logits.view(-1, 2), sub_tlids.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tloss += loss
            # acc
            mask = sub_tlids != -100
            valid_num = mask.to(torch.float).sum().item()
            acc_num = ((logits.max(dim=-1)[1] == sub_tlids) & mask).sum().item()
            acc += acc_num / valid_num
            counter += 1
        tloss /= counter
        return tloss, acc, counter

    def predict(self, batch):
        inpt = batch['ids']
        sids = batch['sids']
        tids = batch['tids']
        mask = batch['mask']   
        lids = batch['tlids']    # [B, S]
        logits = self.model(
            input_ids=inpt,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]    # [B, S, E]
        logits = F.softmax(self.cls(logits), dim=-1)    # [B, S, 2]
        # gather the speacial tokens
        rest = [[] for _ in range(len(inpt))]
        for i in range(len(inpt)):
            index = (lids[i] != -100).to(torch.float).nonzero().squeeze(-1).tolist()    # [2]
            for j in index:
                rest[i].append(logits[i, j, 1])
        rest = torch.stack([torch.stack(i) for i in rest])    # [B, 2]
        # return i->j and j->i
        return rest[:, 0], rest[:, 1]



class BERTCompareV2Retrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareV2Retrieval, self).__init__()
        model = args['pretrained_model']
        self.model = SABertForSequenceClassification.from_pretrained(model, num_labels=2)
        # self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
        # [EOS], [M], [F] for mutual dataset
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)
        self.criterion = nn.CrossEntropyLoss()

        # vocabulary
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')

    def forward(self, batch):
        ids = batch['ids']
        mask = batch['mask']
        tids = batch['tids']
        pids = batch['pids']
        label = batch['label']    # list

        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            compare_ids=pids
        )
        logits = output.logits
        loss = self.criterion(logits, label)
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item() 
        return loss, acc

    def predict(self, batch):
        ids = batch['ids']
        tids = batch['tids']
        pids = batch['pids']
        mask = batch['mask']   
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            compare_ids=pids,
        )
        logits = F.softmax(output.logits, dim=-1)[:, 0]
        return logits



class BERTCompareV3Retrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareV3Retrieval, self).__init__()
        model = args['pretrained_model']
        # self.model = SABertForSequenceClassification.from_pretrained(model, num_labels=3)
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=3)
        # self.model = ElectraForSequenceClassification.from_pretrained(model, num_labels=3)
        # self.model = RobertaForSequenceClassification.from_pretrained(model, num_labels=2)

        # [EOS], [M], [F] for mutual dataset
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)
        self.criterion = nn.CrossEntropyLoss()

        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

    def forward(self, batch):
        ids = batch['ids']
        mask = batch['mask']
        tids = batch['tids']
        pids = batch['pids']
        label = batch['label']    # list

        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            # compare_ids=pids
        )
        logits = output.logits
        loss = self.criterion(logits, label)
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item() 
        return loss, acc

    def predict(self, batch):
        ids = batch['ids']
        tids = batch['tids']
        pids = batch['pids']
        mask = batch['mask']   
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            # compare_ids=pids,
        )
        logits = F.softmax(output.logits, dim=-1)
        # logits = logits[:, 0] - logits[:, 2]
        logits = logits[:, 0]
        return logits



class BERTCompareV4Retrieval(nn.Module):

    def __init__(self, **args):
        super(BERTCompareV4Retrieval, self).__init__()
        model = args['pretrained_model']
        self.model = AutoModel.from_pretrained(model)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size*2, self.model.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(args['dropout']),
            nn.Linear(self.model.config.hidden_size, 3)
        )
        # [EOS], [M], [F] for mutual dataset
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)
        self.criterion = nn.CrossEntropyLoss()

        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]', '[M]', '[F]'])

    def forward(self, batch):
        ids = batch['ids']
        mask = batch['mask']
        tids = batch['tids']
        pids = batch['pids']
        first_index, second_index = batch['first_index'], batch['second_index']
        label = batch['label']    # list
        batch_size = len(ids)

        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            # token_type_ids=tids,
            # compare_ids=pids
        ).last_hidden_state
        hidden = torch.cat((
            output[range(batch_size), first_index, :],
            output[range(batch_size), second_index, :]
        ), dim=-1)    # [B, 2*E]
        logits = self.classifier(hidden)
        loss = self.criterion(logits, label)
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item() 
        return loss, acc

    def predict(self, batch):
        ids = batch['ids']
        tids = batch['tids']
        pids = batch['pids']
        mask = batch['mask']   
        batch_size = len(ids)
        first_index, second_index = batch['first_index'], batch['second_index']
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            # token_type_ids=tids,
            # compare_ids=pids,
        ).last_hidden_state
        hidden = torch.cat((
            output[range(batch_size), first_index, :],
            output[range(batch_size), second_index, :]
        ), dim=-1)    # [B, 2*E]
        logits = self.classifier(hidden)
        logits = F.softmax(logits, dim=-1)[:, 0]
        return logits
