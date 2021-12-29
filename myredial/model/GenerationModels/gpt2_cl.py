from model.utils import *
from dataloader.util_func import *
from inference_utils import *

class GPT2CLEncoder(nn.Module):

    def __init__(self, **args):
        super(GPT2CLEncoder, self).__init__()
        model = args['pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)

        # special tokens
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        faiss_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["model"]}/faiss.ckpt'
        corpus_path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["model"]}/corpus.pkl'
        self.test_max_len = args['test_max_len']
        self.temp = args['temp']
        self.args = args

        # criterion
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

        # model
        self.bert_encoder = BertModel.from_pretrained(args['bert_pretrained_model'])
        self.gpt2_encoder = GPT2LMHeadModel.from_pretrained(args['pretrained_model'])
        self.doc_query_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
        )
        self.token_query_head = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
        )

        # cache index
        if self.args['mode'] in ['test', 'inference']:
            self.cached_index = PhraseSearcher(
                index_type=args['index_type'],
                dimension=args['dimension'],
                nprobe=args['nprobe']
            )
            if self.args['mode'] in ['test']:
                file_prefix = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}'
                self.cached_index.load(f'{file_prefix}/faiss_ckpt.pt', f'{file_prefix}/corpus_ckpt.pt')

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        self.gpt2_encoder.eval()
        gen_logits = self.gpt2_encoder(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    @torch.no_grad()
    def prepare_bert_inputs(self, sentences):
        # check the length
        lengths = [len(b) for b in sentences]
        for l in lengths:
            assert l == self.args['inf_topk']
        ids = []
        for batch in sentences:
            tokens = self.vocab.batch_encode_plus(batch, add_special_tokens=False)['input_ids']
            tokens = [torch.LongTensor([self.cls] + t) for t in tokens]
            ids.extend(tokens)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return ids, ids_mask

    @torch.no_grad()
    def locate_sop(self, bert_ids, bert_ids_mask, gpt2_query, inpt_ids):
        '''locate the start of the phrase'''
        # gpt2_query: [B, E]; bert_hidden: [B*K, S, E]
        bsz, esz = gpt2_query.size()
        _, seqlen = bert_ids.size()
        gpt2_query = self.token_query_head(gpt2_query)

        bert_hidden = self.bert_encoder(input_ids=bert_ids, attention_mask=bert_ids_mask).last_hidden_state[:, 1:, :]
        bert_ids = bert_ids[:, 1:]
        gpt2_query = gpt2_query.unsqueeze(0).expand(self.args['inf_topk'], -1, -1).reshape(-1, esz).unsqueeze(1)    # [B*K, E]
        scores = torch.bmm(gpt2_query, bert_hidden.permute(0, 2, 1)).squeeze(1)    # [B*K, S]
        scores = torch.stack(torch.split(scores, self.args['inf_topk']))    # [B, K, S]
        pos = scores.max(dim=-1)[1]    # [B, K]
        ipdb.set_trace()
    
    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        self.gpt2_encoder.eval()
        self.bert_encoder.eval()
        self.doc_query_head.eval()
        # move the faiss index to the current GPU device    
        self.cached_index.move_to_gpu(device=0)

        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.gpt2_encoder(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            hidden = output.hidden_states[-1][:, -1, :]    # [B, E]
            hidden = self.doc_query_head(hidden)
            
            rest = self.cached_index._search(hidden.cpu().numpy(), topk=self.args['inf_topk'])
            bert_ids, bert_ids_mask = self.prepare_bert_inputs(rest)
            self.locate_sop(bert_ids, bert_ids_mask, hidden, ids)
            
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    @torch.no_grad()
    def _predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        self.gpt2_encoder.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            output = self.gpt2_encoder(
                input_ids=ids,
                attention_mask=ids_mask,
                position_ids=ids_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values
            next_token_logits = logits[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk] = -np.inf
            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(1)    # [B, 1]
            for idx, t in enumerate(next_token.squeeze(-1).tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token
            ids_mask = torch.ones_like(ids)
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    @torch.no_grad()
    def build_offline_index(self, dataloader):
        file_prefix = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}'
        self.gpt2_encoder.eval()
        self.bert_encoder.eval()
        embd, text, effective_index = [], [], []
        pbar = tqdm(dataloader)
        for batch in pbar:
            e = self.bert_encoder(
                input_ids=batch['ids'], 
                attention_mask=batch['ids_mask']
            ).last_hidden_state[:, 0, :]
            embd.append(e)
            text.extend(batch['text'])
            pbar.set_description(f'[!] got document: {len(text)}')
        embd = torch.cat(embd).cpu().numpy()    # [S, E]
        torch.save(
            (embd, text), 
            f'{file_prefix}/faiss_ckpt_{self.args["local_rank"]}.pt',
        )
        # only the first process could build the faiss index
        torch.distributed.barrier()
        if self.args['local_rank'] != 0:
            return 0
        # only the main process will run the following commands
        # load all of the embeddings
        embds, texts = [], []
        for i in tqdm(range(torch.distributed.get_world_size())):
            embd, text = torch.load(f'{file_prefix}/faiss_ckpt_{i}.pt')
            embds.append(embd)
            texts.extend(text)
        embds = np.concatenate(embds)
        assert len(embds) == len(texts)
        self.cached_index._build(embds, texts, speedup=True)
        self.cached_index.save(
            f'{file_prefix}/faiss_ckpt.pt', 
            f'{file_prefix}/corpus_ckpt.pt'
        )
        return len(embds)
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        bert_ids, bert_ids_mask = batch['bert_ids'], batch['bert_ids_mask']
        bsz, seqlen = gpt2_ids.size()

        bert_hidden = self.bert_encoder(bert_ids, bert_ids_mask).last_hidden_state    # [B, S+1, E]
        # with torch.no_grad():
        gpt2_output = self.gpt2_encoder(
            input_ids=gpt2_ids, attention_mask=gpt2_ids_mask, output_hidden_states=True
        )
        gpt2_hidden, gpt2_logits = gpt2_output.hidden_states[-1], gpt2_output.logits

        ## 1. gpt2 mle loss
        # mle loss
        shift_logits = gpt2_logits[..., :-1, :].contiguous()
        shift_labels = gpt2_ids[..., 1:].contiguous()
        mle_loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        mle_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        ## 2. bert doc-level contrastive loss
        # loss: [B, S, E] x [B, 1, E] -> [B, S]
        bert_cls = bert_hidden[:, 0, :]    # [B, E]
        gpt2_hidden_ = self.doc_query_head(gpt2_hidden.permute(1, 0, 2))    # [S, B, E]
        bert_cls = bert_cls.unsqueeze(0).expand(len(gpt2_hidden_), -1, -1)    # [S, B, E]
        logits = torch.bmm(gpt2_hidden_, bert_cls.permute(0, 2, 1))    # [S, B, B]
        logits /= self.temp
        mask = torch.zeros_like(logits)
        mask[:, range(bsz), range(bsz)] = 1.
        loss_ = F.log_softmax(logits, dim=-1) * mask
        doc_level_cl_loss = (-loss_.sum(dim=1)).mean()
        # acc
        doc_level_cl_acc = (logits.max(dim=-1)[1] == torch.arange(bsz).unsqueeze(0).expand(seqlen, -1).cuda()).to(torch.float).mean().item()

        ## 3. bert token-level contrastive loss
        gpt2_hidden = self.token_query_head(gpt2_hidden)    # [B, S, E]
        gpt2_hidden = gpt2_hidden.reshape(-1, 768)
        bert_hidden = bert_hidden[:, 1:, :]    # [B, S, E]
        bert_hidden = bert_hidden.reshape(-1, 768)
        token_level_logits = torch.matmul(gpt2_hidden, bert_hidden.t())    # [B*S, B*S]
        token_level_logits /= self.temp
        mask = torch.zeros_like(token_level_logits)
        mask[range(len(token_level_logits)), range(len(token_level_logits))] = 1.
        loss_ = F.log_softmax(token_level_logits, dim=-1) * mask
        token_level_cl_loss = (-loss_.sum(dim=1)).mean()
        # acc
        token_level_cl_acc = (token_level_logits.max(dim=-1)[1] == torch.arange(len(token_level_logits)).cuda()).to(torch.float).mean().item()

        return (mle_loss, mle_acc), (doc_level_cl_loss, doc_level_cl_acc), (token_level_cl_loss, token_level_cl_acc)
