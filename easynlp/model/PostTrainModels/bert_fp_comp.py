from model.utils import *

class BERTFPCompPostTrain(nn.Module):

    def __init__(self, **args):
        super(BERTFPCompPostTrain, self).__init__()
        model = args['pretrained_model']
        self.model_name = model.lower()
        # self.model = BertForPreTraining.from_pretrained(model)
        # self.model = BertForPreTraining.from_pretrained(model)
        self.model = AutoModelForPreTraining.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+3)    # [EOS]
        if 'roberta' not in model.lower():
            self.model.cls.seq_relationship = nn.Sequential(
                nn.Dropout(p=args['dropout']),
                nn.Linear(self.model.config.hidden_size, 3)
            )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        pids = batch['pids']
        attn_mask = batch['mask']
        mask_labels = batch['mask_labels']
        label = batch['label']

        # [B, S, V]; [B, E]
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            # token_type_ids=token_type_ids,
        )
        try:
            prediction_scores = output.prediction_logits
        except:
            prediction_scores = output.logits

        mlm_loss = self.criterion(
            prediction_scores.view(-1, self.vocab_size),
            mask_labels.view(-1),
        ) 

        if 'roberta' not in self.model_name:
            seq_relationship = output.seq_relationship_logits
            cls_loss = self.criterion(
                seq_relationship.view(-1, 3),
                label.view(-1),
            )
            cls_acc = (seq_relationship.max(dim=-1)[1] == label).to(torch.float).mean().item()
        else:
            cls_loss = torch.tensor(0.).cuda()
            cls_acc = 0.

        # calculate the acc
        not_ignore = mask_labels.ne(-1)
        num_targets = not_ignore.sum().item()
        correct = (prediction_scores.max(dim=-1)[1] == mask_labels) & not_ignore
        correct = correct.sum().item()
        token_acc = correct / num_targets
        return mlm_loss, cls_loss, token_acc, cls_acc
