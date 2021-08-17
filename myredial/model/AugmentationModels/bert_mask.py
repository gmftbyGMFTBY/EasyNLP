from model.utils import *
from dataloader.util_func import *


class BERTMaskAugmentationModel(nn.Module):

    def __init__(self, **args):
        super(BERTMaskAugmentationModel, self).__init__()
        self.args = args
        model = args['pretrained_model']
        self.model = BertForMaskedLM.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        p = args['dropout']
        self.model.cls.seq_relationship = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 3)
        )

        self.vocab = BertTokenizer.from_pretrained(model)
        self.special_tokens = self.vocab.convert_tokens_to_ids(['[PAD]', '[SEP]', '[CLS]'])
        self.mask = self.vocab.convert_tokens_to_ids(['[MASK]'])[0]
        self.pad = self.vocab.convert_tokens_to_ids(['[PAD]'])[0]
        self.da_num = args['augmentation_t']

    @torch.no_grad()
    def forward(self, batch):
        inpt = batch['ids']
        rest = []
        for _ in range(self.da_num):
            ids = []
            for i in deepcopy(inpt):
                mask_sentence_only_mask(i, self.args['min_mask_num'], self.args['max_mask_num'], self.args['masked_lm_prob'], mask=self.mask, vocab_size=len(self.vocab), special_tokens=self.special_tokens)
                i = torch.LongTensor(i)
                ids.append(i)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)

            logits = self.model(
                input_ids=ids,
                attention_mask=mask,
            )[0]    # [B, S, V]
            sent = self.generate_text(ids, F.softmax(logits, dim=-1), batch['response'])    # [B] list
            rest.append(sent)
        # rest: K*[B]
        rest_ = []
        for i in range(len(batch['response'])):
            rest_.append([item[i] for item in rest])
        return rest_

    def generate_text(self, ids, logits, responses):
        sentences = []
        for item, inpt, res in zip(logits, ids, responses):
            inpt = inpt.tolist()
            tokens_ = torch.multinomial(item, num_samples=1).tolist()    # [S, K]
            tokens_ = [token[0] for token in tokens_]
            ts = [t if ot == self.mask else ot for t, ot in zip(tokens_, inpt) if ot not in self.special_tokens and t not in self.special_tokens]
            string = [self.vocab.convert_ids_to_tokens(t) for t in ts]
            string = ''.join(string)
            if string != res:
                sentences.append(string)
            else:
                sentences.append('')
        return sentences