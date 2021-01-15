import os
import torch
from torch import nn
from transformers import BertConfig, BertForPreTraining

class BertDPT(nn.Module):
  
    def __init__(self, hparams):
        super(BertDPT, self).__init__()
        self.hparams = hparams
        self._bert_model = BertForPreTraining.from_pretrained(
            self.hparams.bert_pretrained
        )

    def forward(self, batch):
        bert_outputs = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            masked_lm_labels=batch["masked_lm_labels"],
            next_sentence_label=batch["next_sentence_labels"]
        )
        mlm_loss, nsp_loss, prediction_scores, seq_relationship_score = bert_outputs[:4]

        return None, mlm_loss, nsp_loss