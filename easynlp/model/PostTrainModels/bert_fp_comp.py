from model.utils import *

class BERTFPCompPostTrain(nn.Module):

    def __init__(self, **args):
        super(BERTFPCompPostTrain, self).__init__()
        model = args['pretrained_model']
        self.model = SABertForPreTraining.from_pretrained(model)
        # self.model = BertForPreTraining.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)    # [EOS]
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        # speaker_ids = batch['sids']
        token_type_ids = batch['tids']
        cpids = batch['cpids']
        attn_mask = batch['attn_mask']
        mask_labels = batch['mask_labels']
        label = batch['label']

        # [B, S, V]; [B, E]
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            # speaker_ids=speaker_ids,
            speaker_ids=None,
            compare_ids=cpids
        )
        prediction_scores, seq_relationship = output.prediction_logits, output.seq_relationship_logits

        mlm_loss = self.criterion(
            prediction_scores.view(-1, self.vocab_size),
            mask_labels.view(-1),
        ) 

        cls_loss = self.criterion(
            seq_relationship.view(-1, 2),
            label.view(-1),
        )

        # calculate the acc
        not_ignore = mask_labels.ne(-1)
        num_targets = not_ignore.sum().item()
        correct = (prediction_scores.max(dim=-1)[1] == mask_labels) & not_ignore
        correct = correct.sum().item()
        token_acc = correct / num_targets
        cls_acc = (seq_relationship.max(dim=-1)[1] == label).to(torch.float).mean().item()
        return mlm_loss, cls_loss, token_acc, cls_acc
