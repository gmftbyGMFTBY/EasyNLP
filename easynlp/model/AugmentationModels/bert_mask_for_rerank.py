from model.utils import *
from dataloader.util_func import *


class BERTMaskAugmentationForRerankModel(nn.Module):

    '''will not be trained, must used with GPT2RerankModel'''

    def __init__(self, **args):
        super(BERTMaskAugmentationForRerankModel, self).__init__()
        self.args = args
        model = args['bert_pretrained_model']
        self.model = BertForMaskedLM.from_pretrained(model)
        self.vocab = BertTokenizer.from_pretrained(model)
        self.special_tokens = self.vocab.convert_tokens_to_ids(['[PAD]', '[SEP]', '[CLS]', '[UNK]'])
        self.mask, self.pad = self.vocab.convert_tokens_to_ids(['[MASK]', '[PAD]'])
        self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        # self.ignore_topk = args['ignore_topk']

    @torch.no_grad()
    def forward(self, batch, other_vocab):
        self.model.eval()
        o_ids, ids, label = [], [], []
        for ipt in deepcopy(batch['ids']):
            ipt = [self.cls] + [i for i in ipt.tolist() if i != self.pad] + [self.sep]
            o_ids.append(torch.LongTensor(deepcopy(ipt)))
            mask_label = mask_sentence_only_mask(
                ipt, self.args['min_mask_num'], self.args['max_mask_num'], self.args['masked_lm_prob'], 
                mask=self.mask, vocab_size=len(self.vocab), special_tokens=self.special_tokens
            )
            ids.append(torch.LongTensor(ipt))
            mask_label = torch.LongTensor(mask_label)
            mask_label = torch.where(mask_label == -1, torch.ones_like(mask_label), torch.zeros_like(mask_label))
            label.append(mask_label)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        o_ids = pad_sequence(o_ids, batch_first=True, padding_value=self.pad)
        label = pad_sequence(label, batch_first=True, padding_value=-1)
        mask = generate_mask(ids)
        o_ids, ids, mask, label = to_cuda(o_ids, ids, mask, label)
        logits = self.model(
            input_ids=ids,
            attention_mask=mask,
        )[0]    # [B, S, V]
        ids, ids_mask, label = self.prepare_inputs(o_ids, ids, logits, mask, label, other_vocab)
        # prepare the ids, ids_mask, and label for the GPT2 model
        return ids, ids_mask, label

    @torch.no_grad()
    def prepare_inputs(self, o_ids, ids, logits, mask, label, other_vocab):
        '''ids: [B, S]; logits: [B, S, V]; convert the right padding to left padding for GPT2 model'''
        bsz, seqlen = ids.size()
        rest_ids, rest_label = [], []
        for idx in range(bsz):
            # [S] or [S, V]
            o_ids_, ids_, logits_, mask_, label_ = o_ids[idx], ids[idx], logits[idx], mask[idx], label[idx]
            # ignore the ground-truth tokens
            logits_[range(seqlen), o_ids_] = -np.inf
            # ignore the top-k possible tokens
            # logits_[torch.arange(seqlen).unsqueeze(1), logits_.topk(2, dim=-1)[1]] = -np.inf
            logits_ = F.softmax(logits_, dim=-1)
            n_ids = torch.multinomial(logits_, num_samples=1).squeeze()    # [S]
            n_ids = torch.where(label_ == 0, n_ids, ids_)
            n_label = torch.where(
                n_ids == o_ids_, 
                torch.ones_like(ids_), 
                torch.zeros_like(ids_)
            )    # [S]
            n_label = n_label.masked_fill(label_ == -1, -1)
            # delete the padding and cls/sep tokens and add the padding
            nn_ids, nn_label = [], []
            for i, j in zip(n_ids.tolist(), n_label.tolist()):
                if i not in [self.pad, self.cls, self.sep]:
                    nn_ids.append(i)
                    # avoid too many positive labels
                    if j == 1:
                        if random.random() < 0.2:
                            nn_label.append(j)
                        else:
                            nn_label.append(-1)
                    else:
                        nn_label.append(j)
            rest_label.append(nn_label)
            rest_ids.append(nn_ids)
        # left padding for GPT2 model
        max_length = max([len(i) for i in rest_ids])
        rest_ids = torch.stack([torch.LongTensor([self.pad] * (max_length - len(i)) + i) for i in rest_ids])
        rest_label = torch.stack([torch.LongTensor([-1] * (max_length - len(i)) + i) for i in rest_label])
        rest_ids_mask = generate_mask(rest_ids)
        rest_ids, rest_label, rest_ids_mask = to_cuda(rest_ids, rest_label, rest_ids_mask)
        return rest_ids, rest_ids_mask, rest_label
