from model.utils import *

class BERTMaskAugmentationModel(nn.Module):

    def __init__(self, **args):
        super(BERTMaskAugmentationModel, self).__init__()
        model = args['pretrained_model']
        self.model = BertForMaskedLM.from_pretrained(model)
        self.vocab = BertTokenizer.from_pretrained(model)
        self.mask = self.vocab.convert_tokens_to_ids(['[MASK]'])
        self.da_num = self.args['augmentation_t']

    @torch.no_grad()
    def forward(self, batch):
        inpt = batch['ids']
        attn_mask = batch['mask']
        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
        )[0]    # [B, S, V]
        rest = self.generate_text(F.softmax(logits, dim=-1))
        return rest

    def generate_text(self, ids, logits):
        rest = []
        for _ in range(self.da_num):
            ipdb.set_trace()
            tokens = torch.multinomial(logits, num_samples=1)    # [B, S]
            mask_index = (ids == self.mask).nonzero() 
