from model.utils import *
from dataloader.util_func import *

class WriterELECTRA(nn.Module):

    '''replaced token detection => generated token detection'''

    def __init__(self, **args):
        super(WriterELECTRA, self).__init__()
        model = args['pretrained_model']
        self.model = AutoModel.from_pretrained(args['pretrained_model'])
        self.vocab = AutoTokenizer.from_pretrained(args['pretrained_model'])
        self.cls, self.sep, self.pad = self.vocab.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])
        self.cutoff_length = args['cutoff_length']
        self.cls_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 2)
        )
        self.args = args
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def batchify(self, batch, train=True):
        context, responses = batch['cids'], batch['rids']
        ids, tids, labels = [], [], []
        for idx, (c, rs) in enumerate(zip(context, responses)):
            # collect the candidate for easy negative sampling
            candidates = []
            for i, rr in enumerate(responses):
                if i != idx:
                    candidates.extend(rr)
            # if train:
            #     rs = rs + random.sample(candidates, self.args['easy_cand_num'])
            counter = 0
            for r in rs:
                c_ = deepcopy(c)
                r_ = deepcopy(r)
                truncate_pair(c_, r_, self.args['max_len'])
                ids.append([self.cls] + c_ + [self.sep] + r_ + [self.sep])
                tids.append([0] * (2 + len(c_)) + [1] * (1 + len(r_)))
                real_label = 1 if counter == 0 else 0
                if train:
                    # apply cutoff length
                    former_len = min(2 + len(c_) + self.cutoff_length, 2 + len(c_) + 1 + len(r_))
                    after_len = max(0, 1 + len(r_) - self.cutoff_length)

                    labels.append([-100] * former_len + [real_label] * after_len)
                else:
                    labels.append([-100] * (2 + len(c_)) + [1] * (1 + len(r_)))
                counter += 1
        ids = [torch.LongTensor(i) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = [torch.LongTensor(i) for i in tids]
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        labels = [torch.LongTensor(i) for i in labels]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        ids_mask = generate_mask(ids)
        ids, tids, ids_mask, labels = to_cuda(ids, tids, ids_mask, labels)
        # random shuffle
        random_idx = list(range(len(ids)))
        random.shuffle(random_idx)
        ids = ids[random_idx]
        tids = tids[random_idx]
        ids_mask = ids_mask[random_idx]
        labels = labels[random_idx]
        return ids, tids, ids_mask, labels

    def forward(self, batch):
        ids, tids, ids_mask, labels = self.batchify(batch)
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            token_type_ids=tids,
        )
        logits = self.cls_head(output.last_hidden_state)    # [B, S, 2]
        # random shuffle
        logits, labels = logits.view(-1, 2), labels.view(-1)
        random_idx = torch.randperm(len(logits))
        logits, labels = logits[random_idx], labels[random_idx]

        loss = self.criterion(logits.view(-1, 2), labels.view(-1))

        acc = (logits.view(-1, 2).max(dim=-1)[1] == labels.view(-1)).to(torch.float).mean().item()
        return loss, acc

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        ids, tids, ids_mask, labels = self.batchify(batch, train=False)
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            token_type_ids=tids,
        )
        logits = F.softmax(
            self.cls_head(
                output.last_hidden_state
            ), dim=-1
        )[:, :, 1]    # [B, S]
        labels[labels == -100] = 0
        nums = labels.nonzero(as_tuple=True)[0].tolist()
        effective_num = []
        for i in range(len(ids)):
            effective_num.append(nums.count(i))
        effective = torch.tensor(effective_num).cuda()    # [B]
        logits = torch.where(labels == 0, torch.zeros_like(logits), logits)
        logits = logits.sum(dim=-1)    # [B]
        logits = logits / effective    # [B] 
        return logits
