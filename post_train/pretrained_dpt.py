import os
import ipdb
import torch
from torch import nn
from transformers import BertConfig, BertForPreTraining


class BertNSPMLM(nn.Module):
  
    def __init__(self, hparams):
        super(BertNSPMLM, self).__init__()
        self.hparams = hparams
        self._bert_model = BertForPreTraining.from_pretrained(
            self.hparams['bert_pretrained']
        )

    def forward(self, batch):
        bert_outputs = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["masked_lm_labels"],
            next_sentence_label=batch["next_sentence_labels"].squeeze(-1),    # [B]
        )
        total_loss = bert_outputs[0]
        return total_loss
    
    
class BertMLM(nn.Module):
    
    '''Only have MLM Loss'''
  
    def __init__(self, hparams):
        super(BertMLM, self).__init__()
        self.hparams = hparams
        self._bert_model = BertForMaskedLM.from_pretrained(
            self.hparams['bert_pretrained']
        )

    def forward(self, batch):
        bert_outputs = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["masked_lm_labels"],
        )
        total_loss = bert_outputs[0]
        return total_loss
    
    
class BertNSP(nn.Module):
    
    '''Only have NSP Loss'''
  
    def __init__(self, hparams):
        super(BertNSP, self).__init__()
        self.hparams = hparams
        self._bert_model = BertForNextSentencePrediction.from_pretrained(
            self.hparams['bert_pretrained']
        )

    def forward(self, batch):
        bert_outputs = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["next_sentence_labels"].squeeze(-1),
        )
        total_loss = bert_outputs[0]
        return total_loss