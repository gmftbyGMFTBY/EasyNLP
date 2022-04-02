from model.utils import *
from dataloader.util_func import *
from inference_utils import *

class SimRAGEncoder(nn.Module):

    def __init__(self, **args):
        super(SimRAGEncoder, self).__init__()
        model = args['pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)

        # special tokens
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.test_max_len = args['test_max_len']
        self.args = args
        # criterion
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)
        # model
        self.bert_encoder = BertModel.from_pretrained(args['bert_pretrained_model'])
        self.gpt2_encoder = GPT2LMHeadModel.from_pretrained(args['pretrained_model'])

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
    def predict_topk_topp(self, batch, copy_token_num):
        input_ids = batch['ids']
        _, prefix_length = input_ids.size()
        
        rag_sentence = batch['rag_sentences'][1]
        rag_sentence_ids = torch.LongTensor(self.vocab.encode(rag_sentence, add_special_tokens=False)).cuda()
        copy_token_num = self.copy_detection(input_ids, rag_sentence_ids)
        input_ids = torch.cat([input_ids, rag_sentence_ids[:copy_token_num].unsqueeze(0)], dim=-1)

        beam_output = self.gpt2_encoder.generate(
            input_ids, 
            prefix_length+self.test_max_len, 
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            top_p=self.args['topp'],
            top_k=self.args['topk'],
            do_sample=True,
        )
        beam_output = beam_output[:, prefix_length:]
        return beam_output.tolist()

    @torch.no_grad()
    def copy_detection(self, input_ids, prompt_ids):
        self.gpt2_encoder.eval()
        prefix_length = input_ids.shape[1]
        prompt_ids = prompt_ids[:32]
        ids = torch.cat([input_ids, prompt_ids.unsqueeze(0)], dim=-1)

        output = self.gpt2_encoder(ids)
        logits = output.logits[0, prefix_length-1:-1, :]    # [S, V]
        _, topk_words = logits.topk(self.args['copy_topk'], dim=-1)    # [S, K]
        shift_prompt_ids = prompt_ids.clone()
        assert len(shift_prompt_ids) == len(topk_words) 

        for idx in range(len(shift_prompt_ids)):
            label = shift_prompt_ids[idx].item()
            words = set(topk_words[idx].tolist())    # [K]
            if label in words:
                copy_token_num = idx + 1
                break
        else:
            # fully copy rather than the fully generate?
            copy_token_num = len(shift_prompt_ids)
        return copy_token_num

    @torch.no_grad()
    def predict(self, batch, scorer, copy_token_num):
        '''contrastive search and the token-level rerank'''
        self.gpt2_encoder.eval()
        input_ids = batch['ids']
        prefix_length = len(input_ids[0])

        rag_sentence = batch['rag_sentences'][1]
        rag_sentence_ids = torch.LongTensor(self.vocab.encode(rag_sentence, add_special_tokens=False)).cuda()
        copy_token_num = self.copy_detection(input_ids, rag_sentence_ids)
        input_ids = torch.cat([input_ids, rag_sentence_ids[:copy_token_num].unsqueeze(0)], dim=-1)

        for step in tqdm(range(self.test_max_len)):
            input_ids = PlugAndPlayRAGContrastiveDecodingOneStep(
                self.gpt2_encoder,
                self.vocab,
                scorer,
                scorer.vocab,
                input_ids,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['beta'],
                rag_sentence,
                prefix_length,
            )
        input_ids = input_ids[:, prefix_length:]
        return input_ids.tolist()

    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        gpt2_res_start = batch['s']    # [B]
        gpt2_res_length = batch['l']    # [B]
        bert_ids, bert_ids_mask = batch['bert_ids'], batch['bert_ids_mask']
        bsz, seqlen = gpt2_ids.size()

        bert_cls = self.bert_encoder(bert_ids, bert_ids_mask).last_hidden_state[:, 0, :] 
        with torch.no_grad():
            gpt2_output = self.gpt2_encoder(
                input_ids=gpt2_ids, attention_mask=gpt2_ids_mask, output_hidden_states=True
            )
            gpt2_hidden = gpt2_output.hidden_states[-1]    # [B, S, E]
            gpt2_cls = []
            for item, s, l in zip(gpt2_hidden, gpt2_res_start, gpt2_res_length):
                index = random.choice(list(range(s, s+l)))
                gpt2_cls.append(item[index, :])
            gpt2_cls = torch.stack(gpt2_cls)   # [B, E]
        bert_cls, gpt2_cls = F.normalize(bert_cls, dim=-1), F.normalize(gpt2_cls, dim=-1)

        # contrastive loss
        matrix = torch.matmul(gpt2_cls, bert_cls.t())
        matrix /= self.args['temp']
        mask = torch.zeros_like(matrix)
        mask[range(bsz), range(bsz)] = 1.
        loss_ = F.log_softmax(matrix, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        # acc
        acc = (matrix.max(dim=-1)[1] == torch.arange(bsz).cuda()).to(torch.float).mean().item()
        return loss, acc
