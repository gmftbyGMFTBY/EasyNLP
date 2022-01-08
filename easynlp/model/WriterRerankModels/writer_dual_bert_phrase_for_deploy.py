from model.utils import *
from dataloader.util_func import *
# from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

class WriterPhraseForDeployEncoder(nn.Module):

    '''Phrase-level extraction with GPT-2 LM Head as the query'''

    def __init__(self, **args):
        super(WriterPhraseForDeployEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_pretrained_model']
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.bert_encoder = AutoModel.from_pretrained(model)
        self.gpt2_encoder = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.cls, self.pad, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[PAD]', '[SEP]'])
        self.proj_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 768),        
        )
        self.args = args
    
    def batchify(self, batch):
        '''generate the batch samples for the model'''
        context, responses = batch['cids'], batch['rids']
        gpt2_ids, bert_ids, bert_tids, gpt2_prefix_length, prefix_length, overall_length = [], [], [], [], [], [], []
        for idx, (c, rs) in enumerate(zip(context, responses)):
            for r in rs:
                # bert tokenization
                c_bert = deepcopy(c)
                r_ = deepcopy(r)
                truncate_pair(c_bert, r_, self.args['bert_max_len'])
                bert_ids.append([self.cls] + c_bert + r_ + [self.sep])
                bert_tids.append([0] * (1 + len(c_bert)) + [1] * (1 + len(r_)))
                prefix_length.append(len(c_bert))
                overall_length.append(len(bert_ids[-1]))
            # gpt2 tokenization
            c_ = deepcopy(c)
            gpt2_ids.append([self.cls] + c_ + [self.sep])
            gpt2_prefix_length.append(len(c_))
        gpt2_ids = [torch.LongTensor(i) for i in gpt2_ids]
        bert_ids = [torch.LongTensor(i) for i in bert_ids]
        bert_tids = [torch.LongTensor(i) for i in bert_tids]
        prefix_length = torch.LongTensor(prefix_length)
        gpt2_prefix_length = torch.LongTensor(gpt2_prefix_length)
        overall_length = torch.LongTensor(overall_length)
        gpt2_ids = pad_sequence(gpt2_ids, batch_first=True, padding_value=self.pad)
        bert_ids = pad_sequence(bert_ids, batch_first=True, padding_value=self.pad)
        bert_tids = pad_sequence(bert_tids, batch_first=True, padding_value=self.pad)
        gpt2_ids_mask = generate_mask(gpt2_ids)
        bert_ids_mask = generate_mask(bert_ids)
        gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask = to_cuda(gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask)
        gpt2_prefix_length, prefix_length, overall_length = to_cuda(gpt2_prefix_length, prefix_length, overall_length)
        return {
            'gpt2_ids': gpt2_ids,    # [B, S]
            'gpt2_ids_mask': gpt2_ids_mask,    # [B, S]
            'bert_ids': bert_ids,     # [B*K, S]
            'bert_tids': bert_tids,    # [B*K, S]
            'bert_ids_mask': bert_ids_mask,     # [B*K, S]
            'prefix_length': prefix_length,    # [B*K]
            'gpt2_prefix_length': gpt2_prefix_length,    # [B*K]
            'overall_length': overall_length,    # [B*K]
        }

    def _encode(self, batch):
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
        queries = []
        for item, l in zip(gpt2_rep, batch['gpt2_prefix_length']):
            queries.append(item[l])    # [E]
        queries = torch.stack(queries)    # [B, E]
        # collect the embedings
        embeddings = []
        for item, l, ol in zip(bert_rep, batch['prefix_length'], batch['overall_length']):
            # item: [S, E], ignore the mask tokens, and calculate the average embeddings
            embeddings.append(item[l:ol, :].mean(dim=0))
        embeddings = torch.stack(embeddings)    # [B*K, E]
        # queries: [B, E]; embeddings: [B*K, E]
        return queries, embeddings

    def forward(self, 
        gpt2_ids, 
        gpt2_ids_mask,
        bert_ids,
        bert_tids,
        bert_ids_mask,
        prefix_length,
        gpt2_prefix_length,
        overall_length
    ):
        self.gpt2_encoder.eval()
        self.bert_encoder.eval()
        batch = {
            'gpt2_ids': gpt2_ids,
            'gpt2_ids_mask': gpt2_ids_mask,
            'bert_ids': bert_ids,
            'bert_tids': bert_tids,
            'bert_ids_mask': bert_ids_mask,
            'prefix_length': prefix_length,
            'gpt2_prefix_length': gpt2_prefix_length,
            'overall_length': overall_length,
        }
        queries, embeddings = self._encode(batch)
        dot_product = torch.matmul(queries, embeddings.t()).squeeze(0)
        # api normalization
        dot_product /= np.sqrt(768)
        dot_product = (dot_product - dot_product.min()) / (1e-3 + dot_product.max() - dot_product.min())
        return dot_product

    def convert_to_onnx(self, output_path):
        self.gpt2_encoder.eval()
        self.bert_encoder.eval()
        # prepare the input
        batch_size = 32
        gpt2_seq_len = 64
        bert_seq_len = 128
        gpt2_ids = torch.ones(batch_size, gpt2_seq_len).to(torch.long)
        gpt2_ids_mask = torch.ones(batch_size, gpt2_seq_len).to(torch.long)
        bert_ids = torch.ones(batch_size, bert_seq_len).to(torch.long)
        bert_tids = torch.ones(batch_size, bert_seq_len).to(torch.long)
        bert_ids_mask = torch.ones(batch_size, bert_seq_len).to(torch.long)
        prefix_length = torch.ones(batch_size).to(torch.long)
        gpt2_prefix_length = torch.ones(batch_size).to(torch.long)
        overall_length = torch.ones(batch_size).to(torch.long)
        gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask, prefix_length, gpt2_prefix_length, overall_length = to_cuda(gpt2_ids, gpt2_ids_mask, bert_ids, bert_tids, bert_ids_mask, prefix_length, gpt2_prefix_length, overall_length)

        torch.onnx.export(
            model=self,
            args=(
                gpt2_ids,
                gpt2_ids_mask,
                bert_ids,
                bert_tids,
                bert_ids_mask,
                prefix_length,
                gpt2_prefix_length,
                overall_length,
            ),
            f=output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=[
                'gpt2_ids', 
                'gpt2_ids_mask',
                'bert_ids',
                'bert_tids',
                'bert_ids_mask',
                'prefix_length',
                'gpt2_prefix_length',
                'overall_length'
            ],
            output_names=["predictions"],
            dynamic_axes={
                'gpt2_ids': {0: 'batch_size', 1: "seq_len"},
                'gpt2_ids_mask': {0: 'batch_size', 1: "seq_len"},
                'bert_ids': {0: 'batch_size', 1: "seq_len"},
                'bert_tids': {0: 'batch_size', 1: "seq_len"},
                'bert_ids_mask': {0: 'batch_size', 1: "seq_len"},
                'prefix_length': {0: 'batch_size'},
                'gpt2_prefix_length': {0: 'batch_size'},
                'overall_length': {0: 'batch_size'},
                "predictions": {0: 'batch_size'}
            }
        )
        optimize_by_onnxruntime(
            onnx_model_path=output_path,
            use_gpu=True,
            optimized_model_path=output_path,
        )
