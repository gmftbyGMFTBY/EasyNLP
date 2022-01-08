from model.utils import *

class WriterDualCompareEncoder(nn.Module):

    def __init__(self, **args):
        super(WriterDualCompareEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_pretrained_model']
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.bert_encoder = AutoModel.from_pretrained(model)
        self.gpt2_encoder = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.cls, self.pad, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[PAD]', '[SEP]'])
        self.topk = 1 + args['inference_time']
        # compare encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=args['nhead'])
        self.fusion_encoder = nn.TransformerDecoder(decoder_layer, num_layers=args['num_layers'])
        self.args = args
    
    def batchify(self, batch, train=True, deploy=False):
        context, responses = batch['cids'], batch['rids']
        gpt2_ids, bert_ids, bert_tids, gpt2_prefix_length = [], [], [], []
        for idx, (c, rs) in enumerate(zip(context, responses)):
            if train is False and deploy is False:
                # test model (not deploy mode)
                rs = rs + batch['erids']
            for r in rs:
                c_bert = deepcopy(c)
                r_ = deepcopy(r)
                truncate_pair(c_bert, r_, self.args['bert_max_len'])
                bert_ids.append([self.cls] + c_bert + r_ + [self.sep])
                bert_tids.append([0] * (1 + len(c_bert)) + [1] * (1 + len(r_)))
            # gpt2 tokenization
            c_ = deepcopy(c)
            gpt2_ids.append(c_)
            gpt2_prefix_length.append(len(c_)-1)
        gpt2_ids = [torch.LongTensor(i) for i in gpt2_ids]
        bert_ids = [torch.LongTensor(i) for i in bert_ids]
        bert_tids = [torch.LongTensor(i) for i in bert_tids]
        gpt2_prefix_length = torch.LongTensor(gpt2_prefix_length)
        gpt2_ids = pad_sequence(gpt2_ids, batch_first=True, padding_value=self.pad)
        bert_ids = pad_sequence(bert_ids, batch_first=True, padding_value=self.pad)
        bert_tids = pad_sequence(bert_tids, batch_first=True, padding_value=self.pad)
        gpt2_ids_mask = generate_mask(gpt2_ids)
        bert_ids_mask = generate_mask(bert_ids)
        gpt2_prefix_length, gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask = to_cuda(gpt2_prefix_length, gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask)
        return {
            'gpt2_ids': gpt2_ids,    # [B, S]
            'gpt2_ids_mask': gpt2_ids_mask,    # [B, S]
            'bert_ids': bert_ids,     # [B*K, S]
            'bert_tids': bert_tids,    # [B*K, S]
            'bert_ids_mask': bert_ids_mask,     # [B*K, S]
            'gpt2_prefix_length': gpt2_prefix_length,    # [B*K]
        }

    def _encode(self, batch, train=True):
        gpt2_rep_whole = self.gpt2_encoder(
            input_ids=batch['gpt2_ids'], 
            attention_mask=batch['gpt2_ids_mask'],
            output_hidden_states=True,
        ).hidden_states[-1]    # [B_c, S, E]
        gpt2_rep = gpt2_rep_whole[range(len(gpt2_rep_whole)), batch['gpt2_prefix_length'], :]    # [B_c, E]
        # bert encoder 
        bert_rep = self.bert_encoder(
            input_ids=batch['bert_ids'],
            token_type_ids=batch['bert_tids'],
            attention_mask=batch['bert_ids_mask'],
        ).last_hidden_state[:, 0, :]    # [B_r*K, E]

        # prepare inputs for comparison
        if train is True:
            bert_rep_ = torch.stack(torch.split(bert_rep, self.topk))[:, 0, :]    # [B_r, E]
        else:
            bert_rep_ = bert_rep.clone()
        rep_rid = bert_rep_.unsqueeze(1).expand(-1, len(gpt2_rep), -1)
        rest = self.fusion_encoder(
            rep_rid,
            gpt2_rep_whole,
            memory_key_padding_mask=~batch['gpt2_ids_mask'].to(torch.bool),
        )    # [B_r, B_c, E]
        rest = F.normalize(rest, dim=-1)
        # normalization
        dp = torch.bmm(
            F.normalize(gpt2_rep.unsqueeze(1), dim=-1), 
            rest.permute(1, 2, 0)
        ).squeeze(1)    # [B_c, B_r*K]
        if train is False:
            return dp   # [B_c, B_r*K]

        # hard negative with random negative
        rep_rid = bert_rep.unsqueeze(1).expand(-1, len(gpt2_rep), -1)
        rest = self.fusion_encoder(
            rep_rid,
            gpt2_rep_whole,
            memory_key_padding_mask=~batch['gpt2_ids_mask'].to(torch.bool),
        )    # [B_r*K, B_c, E]
        rest = F.normalize(rest, dim=-1)
        # normalization
        dp3 = torch.bmm(
            F.normalize(gpt2_rep.unsqueeze(1), dim=-1), 
            rest.permute(1, 2, 0)
        ).squeeze(1)    # [B_c, B_r*K]

        # hard negative comparison
        rep_rid = torch.stack(torch.split(bert_rep, self.topk)).permute(1, 0, 2)   # [K, B_r, E]
        rest = self.fusion_encoder(
            rep_rid,
            gpt2_rep_whole,
            memory_key_padding_mask=~batch['gpt2_ids_mask'].to(torch.bool),
        )    # [K, B_r, E]
        rest = F.normalize(rest, dim=-1)
        dp2 = torch.bmm(
            F.normalize(gpt2_rep.unsqueeze(1)), 
            rest.permute(1, 2, 0)
        ).squeeze(1)    # [B_c, K]
        return dp, dp2, dp3

    @torch.no_grad()
    def predict(self, batch):
        self.bert_encoder.eval()
        self.gpt2_encoder.eval()
        deploy = batch['deploy'] if 'deploy' in batch else False
        batch = self.batchify(batch, train=False, deploy=deploy)
        dp = self._encode(batch, train=False)    # [B_c, B_r*K]
        # for the case that its batch size is 1
        return F.softmax(dp, dim=-1)[0]
    
    def forward(self, batch):
        batch = self.batchify(batch)
        # dp1: [B_c, B_r*K]; [B_c, K]
        dp1, dp2, dp3 = self._encode(batch)
        dp1 /= self.args['temp']
        dp2 /= self.args['temp']
        dp3 /= self.args['temp']
        # loss 1: mask the hard negative
        mask = torch.zeros_like(dp1)
        mask[range(len(dp1)), range(len(dp1))] = 1.
        loss_ = F.log_softmax(dp1, dim=-1) * mask
        loss1 = (-loss_.sum(dim=1)).mean()
        acc1 = (dp1.max(dim=-1)[1].cpu() == torch.arange(len(dp1))).to(torch.float).mean().item()
        # loss 2
        mask = torch.zeros_like(dp2)
        mask[:, 0] = 1.
        loss_ = F.log_softmax(dp2, dim=-1) * mask
        loss2 = (-loss_.sum(dim=1)).mean()
        acc2 = (dp2.max(dim=-1)[1].cpu() == torch.zeros(len(dp2))).to(torch.float).mean().item()
        # loss 3: hard negative with some random negative
        s1, s2 = dp3.size()
        mask = torch.zeros_like(dp3)
        mask[range(s1), range(0, s2, self.topk)] = 1.
        loss_ = F.log_softmax(dp3, dim=-1) * mask
        loss3 = (-loss_.sum(dim=1)).mean()
        acc3  = (dp3.max(dim=-1)[1].cpu() == torch.arange(0, s2, self.topk)).to(torch.float).mean().item()
        return loss1, loss2, loss3, acc1, acc2, acc3
