from model.utils import *

class WriterDualCompareEncoder(nn.Module):

    '''Phrase-level extraction with GPT-2 LM Head as the query'''

    def __init__(self, **args):
        super(WriterDualCompareEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_pretrained_model']
        # vocab
        self.vocab = AutoTokenizer.from_pretrained(model)
        # model
        self.bert_encoder = AutoModel.from_pretrained(model)
        self.gpt2_encoder = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.cls, self.pad, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[PAD]', '[SEP]'])
        self.proj_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 768),        
        )
        self.topk = (1 + args['easy_cand_num'] + args['inference_time']) * (1 + args['additional_embeddings'])
        self.inner_topk = 1 + args['easy_cand_num'] + args['inference_time']
        self.args = args
    
    def batchify(self, batch, train=True, deploy=False):
        context, responses = batch['cids'], batch['rids']
        gpt2_ids, bert_ids, bert_tids, labels, gpt2_prefix_length, prefix_length, overall_length = [], [], [], [], [], [], []
        for idx, (c, rs) in enumerate(zip(context, responses)):
            # collect easy negative samples
            if train:
                # collect the in-batch easy negative samples
                candidates = []
                for i, rr in enumerate(responses):
                    if i != idx:
                        candidates.extend(rr)
                if self.args['easy_cand_num'] > len(candidates):
                    # use the easy cand pool
                    rs = rs + random.sample(batch['erids'], self.args['easy_cand_num'])
                else:
                    rs = rs + random.sample(candidates, self.args['easy_cand_num'])
            elif deploy is False:
                rs = rs + batch['erids']

            for r in rs:
                # bert tokenization
                c_bert = deepcopy(c)
                r_ = deepcopy(r)
                truncate_pair(c_bert, r_, self.args['bert_max_len'])
                bert_ids.append([self.cls] + c_bert + r_ + [self.sep])
                bert_tids.append([0] * (1 + len(c_bert)) + [1] * (1 + len(r_)))
                prefix_length.append(len(c_bert))
                overall_length.append(len(bert_ids[-1]))
            labels.extend([1] + [0] * (len(rs) - 1))
            # gpt2 tokenization
            c_ = deepcopy(c)
            gpt2_ids.append([self.cls] + c_ + [self.sep])
            gpt2_prefix_length.append(len(c_))
        gpt2_ids = [torch.LongTensor(i) for i in gpt2_ids]
        bert_ids = [torch.LongTensor(i) for i in bert_ids]
        bert_tids = [torch.LongTensor(i) for i in bert_tids]
        labels = torch.LongTensor(labels)
        prefix_length = torch.LongTensor(prefix_length)
        gpt2_prefix_length = torch.LongTensor(gpt2_prefix_length)
        overall_length = torch.LongTensor(overall_length)
        gpt2_ids = pad_sequence(gpt2_ids, batch_first=True, padding_value=self.pad)
        bert_ids = pad_sequence(bert_ids, batch_first=True, padding_value=self.pad)
        bert_tids = pad_sequence(bert_tids, batch_first=True, padding_value=self.pad)
        gpt2_ids_mask = generate_mask(gpt2_ids)
        bert_ids_mask = generate_mask(bert_ids)
        gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask, labels = to_cuda(gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask, labels)
        gpt2_prefix_length, prefix_length, overall_length = to_cuda(gpt2_prefix_length, prefix_length, overall_length)
        return {
            'gpt2_ids': gpt2_ids,    # [B, S]
            'gpt2_ids_mask': gpt2_ids_mask,    # [B, S]
            'bert_ids': bert_ids,     # [B*K, S]
            'bert_tids': bert_tids,    # [B*K, S]
            'bert_ids_mask': bert_ids_mask,     # [B*K, S]
            'labels': labels,    # [B*K]
            'prefix_length': prefix_length,    # [B*K]
            'gpt2_prefix_length': gpt2_prefix_length,    # [B*K]
            'overall_length': overall_length,    # [B*K]
        }

    def _encode(self, batch, train=True):
        with torch.no_grad():
            # donot train the GPT-2 model
            gpt2_rep = self.gpt2_encoder(
                input_ids=batch['gpt2_ids'], 
                attention_mask=batch['gpt2_ids_mask'],
                output_hidden_states=True,
            ).hidden_states[-1]
        gpt2_rep = self.proj_head(gpt2_rep)
        # bert encoder 
        bert_rep = self.bert_encoder(
            input_ids=batch['bert_ids'],
            token_type_ids=batch['bert_tids'],
            attention_mask=batch['bert_ids_mask'],
        ).last_hidden_state
        # gpt2_rep, bert_rep: [B, S, E], [B*K, S, E]
        # collect the queries
        # queries = []
        # for item, l in zip(gpt2_rep, batch['gpt2_prefix_length']):
        #     queries.append(item[l])    # [E]
        # queries = torch.stack(queries)    # [B, E]
        queries = gpt2_rep[range(len(gpt2_rep)), batch['gpt2_prefix_length'], :]    # [B, E]
        # collect the embedings
        embeddings = []
        for item, l, ol in zip(bert_rep, batch['prefix_length'], batch['overall_length']):
            # item: [S, E], ignore the mask tokens, and calculate the average embeddings
            embeddings.append(item[l:ol, :].mean(dim=0))
            # add the additional embeddings
            if train:
                for _ in range(self.args['additional_embeddings']):
                    begin = random.randint(0, l-int(l*0.1))
                    end   = random.randint(begin + 1, ol) 
                    embeddings.append(item[begin:end, :].mean(dim=0))
        embeddings = torch.stack(embeddings)    # [B*K, E]
        # queries: [B, E]; embeddings: [B*K, E]
        return queries, embeddings

    @torch.no_grad()
    def predict(self, batch):
        deploy = batch['deploy'] if 'deploy' in batch else False
        batch = self.batchify(batch, train=False, deploy=deploy)
        queries, embeddings = self._encode(batch, train=False)
        dot_product = torch.matmul(queries, embeddings.t()).squeeze(0)    # [20]
        # api normalization
        dot_product = (dot_product + 1)/2
        return dot_product
    
    def forward(self, batch):
        batch = self.batchify(batch)
        # convert text to embeddings
        # queries: [B, E]; embeddings: [B*K*10, E]
        queries, embeddings = self._encode(batch)
        dot_product = torch.matmul(queries, embeddings.t())
        s1, s2 = dot_product.size()
        # loss 1
        mask = torch.zeros_like(dot_product)
        mask[range(s1), range(0, s2, self.topk)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (dot_product.max(dim=-1)[1].cpu() == torch.arange(0, s2, self.topk)).to(torch.float).mean().item()
        # loss 2: generated responses better than the random negative samples
        # mask the ground-truth
        dot_product_ = torch.where(
            torch.zeros_like(dot_product).to(torch.bool), 
            dot_product, 
            dot_product
        )
        dot_product_[range(s1), range(0, s2, self.topk)] = -1000
        # collect the index of the inferenced samples
        inference_gt_idx = [[] for _ in range(s1)]
        for batch_i in range(s1):
            i_idx = [self.topk * batch_i + self.inner_topk * (1 + j) for j in range(self.args['inference_time'])]
            inference_gt_idx[batch_i].extend(i_idx)
        loss2 = 0
        for i_i in range(self.args['inference_time']):
            # mask other inference samples, only use one as positive
            dot_product_bck = torch.where(
                torch.zeros_like(dot_product_).to(torch.bool), 
                dot_product_, 
                dot_product_
            )
            for_i_idx = []
            for batch_i in range(s1):
                i_i_ = self.topk * batch_i + self.inner_topk * (1 + i_i)
                mask_inference_index = list(set(inference_gt_idx[batch_i]) - set([i_i_]))
                dot_product_bck[batch_i, mask_inference_index] = -1000
                for_i_idx.append(i_i_)
            # calculate the loss
            mask = torch.zeros_like(dot_product_bck)
            mask[range(s1), for_i_idx] = 1.
            loss_ = F.log_softmax(dot_product_bck, dim=-1) * mask
            loss2 += (-loss_.sum(dim=1)).mean()
        loss2 /= self.args['inference_time']
        loss = loss + loss2
        return loss, acc
