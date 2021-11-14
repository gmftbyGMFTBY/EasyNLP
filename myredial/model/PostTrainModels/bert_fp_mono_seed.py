from model.utils import *

class BERTMonoSEEDPostTrain(nn.Module):

    '''use the weight of the bert-fp, and only the masked lm loss is used'''

    def __init__(self, **args):
        super(BERTMonoSEEDPostTrain, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']

        self.model = BertForPreTraining.from_pretrained(model)
        self.decoder = WeakTrsDecoder(
            args['dropout'] ,
            args['vocab_size'],
            args['nhead'],
            args['nlayer'],
            args['attention_span']
        )
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)    # [EOS]
        self.model.cls.seq_relationship = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 3)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        no_mask_inpt = batch['no_mask_ids']
        attn_mask = batch['attn_mask']
        mask_labels = batch['mask_labels']

        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        prediction_scores = output.prediction_logits
        mlm_loss = self.criterion(
            prediction_scores.view(-1, self.vocab_size),
            mask_labels.view(-1),
        ) 

        # weak decoder loss and acc
        de_loss, de_acc = self.decoder(
            no_mask_inpt, 
            output.hidden_states[-1].permute(1, 0, 2)[:1, :, :],
            attn_mask,
        )

        # calculate the acc
        not_ignore = mask_labels.ne(-1)
        num_targets = not_ignore.sum().item()
        correct = (prediction_scores.max(dim=-1)[1] == mask_labels) & not_ignore
        correct = correct.sum().item()
        token_acc = correct / num_targets
        return mlm_loss, de_loss, token_acc, de_acc
