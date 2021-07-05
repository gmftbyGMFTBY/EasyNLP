from model.utils import *

class BERTFPPostTrain(nn.Module):

    '''2: right, 1: within session; 0: random'''

    def __init__(self, **args):
        super(BERTFPPostTrain, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']

        self.model = BertForPreTraining.from_pretrained(model)
        self.model.cls.seq_relationship = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 3)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['attn_mask']
        mask_labels = batch['mask_labels']
        label = batch['label']

        # [B, S, V]; [B, E]
        prediction_scores, seq_relationship = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )

        mlm_loss = self.criterion(
            prediction_scores.view(-1, self.vocab_size),
            mask_labels.view(-1),
        ) 

        cls_loss = self.criterion(
            seq_relationship,view(-1, 3),
            label.view(-1),
        )

        # calculate the acc
        token_acc = (prediction_scores.view(-1, self.vocab_size) == mask_labels.view(-1)).mean().item()
        cls_acc = (seq_relationship == label).mean().item()

        return mlm_loss, cls_loss, token_acc, cls_acc
