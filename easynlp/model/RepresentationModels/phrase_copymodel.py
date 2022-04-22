from model.utils import *

class DensePhraseEncoder(nn.Module):

    '''contrastive search for gpt2 model.
    For inference, please load the model into the gpt2 model (model_name in the GenerationModels)'''

    def __init__(self, **args):
        super(DensePhraseEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # bert-encoder model
        self.phrase_encoder = BertModel.from_pretrained(self.args['phrase_encoder_model'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args['phrase_tokenizer'])
        self.pad = self.bert_tokenizer.pad_token_id
        self.sep = self.bert_tokenizer.sep_token_id
        self.hn_num = self.args['hard_neg_for_each_doc']

        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2Model.from_pretrained(model_name)
        self.s_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(768, 768)        
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=self.args['dropout']),
            nn.Tanh(),
            nn.Linear(768, 768)        
        )

    @torch.no_grad()
    def get_phrase_rep(self, ids, ids_mask, pos, text):
        self.eval()
        rep = self.phrase_encoder(ids, ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        phrases, texts = [], []
        for rep_, pos_, text_ in zip(rep, pos, text):
            for (b, e) in pos_:
                b_rep = rep_[b, :]
                e_rep = rep_[e, :]
                p_rep = torch.cat([b_rep, e_rep], dim=-1)
                phrases.append(p_rep)
            texts.extend(text_)
        phrases = torch.stack(phrases)
        phrases = F.normalize(phrases, dim=-1)
        return phrases, texts

    @torch.no_grad()
    def get_query_rep(self, ids):
        rep = self.model(input_ids=ids).last_hidden_state
        query = torch.cat([
            self.s_proj(rep[:, -1, :]),
            self.e_proj(rep[:, -1, :])
        ], dim=-1)
        query = F.normalize(query, dim=-1)
        return query

    def forward(self, batch):
        ## generation fine-tuning gpt2
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask)
        last_hidden_states = outputs.last_hidden_state

        # phrase begin position
        pos_index = batch['pos_ids']    # [B_p]
        ext_last_hidden_states = []
        for hs, pos_list in zip(last_hidden_states, pos_index):
            ext_last_hidden_states.append(hs[pos_list, :])
        query_rep = torch.cat(ext_last_hidden_states)    # [B_q]
        query_rep = torch.cat(
            [self.s_proj(query_rep), self.e_proj(query_rep)], dim=-1        # [B_p, 2*E]
        )

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']

        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        doc_bsz, seqlen, _ = output.size()
        # extract the phrase representations
        s_rep = output[range(doc_bsz), dindex_s, :]    # [B, E]
        e_rep = output[range(doc_bsz), dindex_e, :]

        # hard phrase rep
        effective_num = dids_mask.sum(dim=-1)    # [B]
        hn_s_rep, hn_e_rep = [], []
        for o, m, s_pos, e_pos in zip(output, effective_num, dindex_s, dindex_e):
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(s_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_s_rep.append(o[effective_range, :])
            
            effective_range = list(range(1, m))   # ignore the [CLS] token
            effective_range.remove(e_pos)
            if len(effective_range) > self.hn_num:
                effective_range = random.sample(effective_range, self.hn_num)
            hn_e_rep.append(o[effective_range, :])
        hn_s_rep = torch.cat(hn_s_rep)
        hn_e_rep = torch.cat(hn_e_rep)
        
        # phrase rep consists of the in-batch negative and in-doc negative samples
        phrase_rep = torch.cat([s_rep, e_rep], dim=-1)    # [B_p, 2*E]
        hn_phrase_rep = torch.cat([hn_s_rep, hn_e_rep], dim=-1)    # [B_hn, 2*E]
        phrase_rep = torch.cat([phrase_rep, hn_phrase_rep], dim=0)    # [B_p, 2*E]

        query_rep, phrase_rep = F.normalize(query_rep, dim=-1), F.normalize(phrase_rep, dim=-1)
        # distributed collection
        # query_rep, phrase_rep = distributed_collect(query_rep, phrase_rep)
        # doc_bsz = len(query_rep)
        
        dp = torch.matmul(query_rep, phrase_rep.t())    # [B_p, B_p]
        dp /= self.args['temp']
        mask = torch.zeros_like(dp)
        mask[range(doc_bsz), range(doc_bsz)] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        phrase_loss = (-loss_.sum(dim=1)).mean()
        phrase_acc = (dp.max(dim=-1)[1].cpu() == torch.LongTensor(torch.arange(doc_bsz))).to(torch.float).mean().item()
        return phrase_loss, phrase_acc
