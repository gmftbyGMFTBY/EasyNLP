from model.utils import *
from model.GenerationModels.utils import *

class DialogEVA(nn.Module):

    def __init__(self, **args):
        super(DialogEVA, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.special_tokens = set([self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        self.pad = self.tokenizer.pad_token_id
        self.cls = self.tokenizer.cls_token_id
        self.unk = self.tokenizer.unk_token_id
        self.sep = self.tokenizer.sep_token_id
        self.vocab_size = len(self.tokenizer)

        self.topp = self.args['topp']
        self.topk = self.args['topk']

        # model
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.embed_dim = self.model.config.hidden_size
        self.test_max_len = args['test_gen_max_len']
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
    
    @torch.no_grad()
    def predict(self, context_list):
        self.model.eval()
        items = self.tokenizer.batch_encode_plus(context_list, add_special_tokens=False)['input_ids']
        # convert string to tokens
        context_ids = []
        for u in items:
            context_ids.extend(u + [self.sep])
        context_ids.pop()
        ids = [self.cls] + context_ids[-self.args['max_prefix_len']:] + [self.sep]
        ids = torch.LongTensor(ids).unsqueeze(0).cuda()
        beam_output = self.model.generate(
            ids,
            self.test_max_len,
            pad_token_id=self.pad,
            eos_token_id=self.sep,
            top_p=self.topp,
            top_k=self.topk,
            do_sample=True
        )
        beam_output = beam_output[0]
        string = ''.join(self.tokenizer.convert_ids_to_tokens(beam_output.tolist())).replace('[SEP]', '')
        return string

    def forward(self, batch):
        input_ids, input_ids_mask = batch['input_ids'], batch['input_ids_mask']
        output_ids, output_ids_mask, labels = batch['output_ids'], batch['output_ids_mask'], batch['labels']
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=input_ids_mask, 
            decoder_input_ids=output_ids,
            decoder_attention_mask=output_ids_mask,
        )
        logits = outputs.logits
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        # token_acc
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.view(-1) == labels.view(-1)).to(torch.long)
        valid_mask = (labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return mle_loss, gen_acc

    def calculate_ppl(self, input_ids, ids_mask, pos_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask, position_ids=pos_ids)
        logits = outputs.logits
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        return math.exp(mle_loss.item())
