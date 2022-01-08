from model.utils import *
from dataloader.util_func import *

class WriterBERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(WriterBERTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = AutoModel.from_pretrained(args['pretrained_model'])
        self.vocab = AutoTokenizer.from_pretrained(args['pretrained_model'])
        self.cls, self.sep, self.pad = self.vocab.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])
        self.cls_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 2)
        )
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def batchify(self, batch, train=True):
        context, responses = batch['cids'], batch['rids']
        ids, tids, label = [], [], []
        for idx, (c, rs) in enumerate(zip(context, responses)):
            # collect the candidate for easy negative sampling
            candidates = []
            for i, rr in enumerate(responses):
                if i != idx:
                    candidates.extend(rr)
            if train:
                rs = rs + random.sample(candidates, self.args['easy_cand_num'])
            for r in rs:
                c_ = deepcopy(c)
                r_ = deepcopy(r)
                truncate_pair(c_, r_, self.args['max_len'])
                ids.append([self.cls] + c_ + [self.sep] + r_ + [self.sep])
                tids.append([0] * (2 + len(c_)) + [1] * (1 + len(r_)))
            label.extend([1] + [0] * (len(rs) - 1))
        label = torch.LongTensor(label)
        ids = [torch.LongTensor(i) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = [torch.LongTensor(i) for i in tids]
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, tids, ids_mask, label = to_cuda(ids, tids, ids_mask, label)
        # random shuffle
        random_idx = list(range(len(label)))
        random.shuffle(random_idx)
        ids = ids[random_idx]
        tids = tids[random_idx]
        ids_mask = ids_mask[random_idx]
        label = label[random_idx]
        return ids, tids, ids_mask, label

    def forward(self, batch):
        ids, tids, ids_mask, label = self.batchify(batch)
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            token_type_ids=tids,
        )
        logits = self.cls_head(output.last_hidden_state[:, 0, :])    # [B, 2]
        loss = self.criterion(logits, label)
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        return loss, acc

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        ids, tids, ids_mask, label = self.batchify(batch, train=False)
        output = self.model(
            input_ids=ids,
            attention_mask=ids_mask,
            token_type_ids=tids,
        )
        logits = F.softmax(
            self.cls_head(
                output.last_hidden_state[:, 0, :]
            ), dim=-1
        )    # [B, 2]
        return logits[:, 1]
