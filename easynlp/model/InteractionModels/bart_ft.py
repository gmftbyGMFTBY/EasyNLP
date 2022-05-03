from model.utils import *

class BARTRetrieval(nn.Module):

    def __init__(self, **args):
        super(BARTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BartModel.from_pretrained(model, num_labels=2)
        self.vocab = BertTokenizer.from_pretrained(args['tokenizer'])
        self.cls = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 2)
        )

    def forward(self, batch):
        inpt = batch['ids']
        attn_mask = batch['ids_mask']
        opt = batch['rids']
        opt_attn_mask = batch['rids_mask']

        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            decoder_input_ids=opt,
            decoder_attention_mask=opt_attn_mask
        )
        decoder_length = opt_attn_mask.sum(dim=-1) - 1    # [B]
        bsz, _ = inpt.size()
        hidden_state = self.cls(output.last_hidden_state[range(bsz), decoder_length, :])    # [B, S, 2]
        return hidden_state
