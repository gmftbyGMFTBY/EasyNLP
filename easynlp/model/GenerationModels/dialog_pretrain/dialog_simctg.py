from model.utils import *
from model.GenerationModels.utils import *

class DialogSimCTG(nn.Module):

    def __init__(self, **args):
        super(DialogSimCTG, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_tokens = set([self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        self.pad = self.tokenizer.pad_token_id
        self.unk = self.tokenizer.unk_token_id
        self.sep = self.tokenizer.sep_token_id
        self.vocab_size = len(self.tokenizer)

        # model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.embed_dim = self.model.config.hidden_size
        # decoding length
        self.test_max_len = args['test_gen_max_len']
        # ignore the pad_token (-100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.margin = args['margin']
        self.topk = self.args['contrastive_topk']
        self.topp = self.args['contrastive_topp']
    
    @torch.no_grad()
    def predict(self, batch):
        '''contrastive search'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]

        past_key_values = None
        last_hidden_states = None
        first_step = 0
        logits = None
        for step in range(self.test_max_len):
            ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepBatch(
                self.model,
                ids,
                ids_mask,
                ids_pos,
                self.args['beam_width'],
                self.args['model_prediction_confidence'],
                self.args['contrastive_topk'],
                self.args['contrastive_topp'],
                self.pad,
                min(1., (step+1)/self.args['sep_smooth_length']),
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                step,
                step < self.args['sampling_prefix_len'],
            )
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
            ids_mask = torch.ones_like(ids)
            # collect ids: [B, 1]
            tokens = ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
        # ignore the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    def forward(self, batch):
        input_ids, ids_mask, labels = batch['ids'], batch['ids_mask'], batch['ods']
        ipdb.set_trace()
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask, output_hidden_states=True)
        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        # token_acc
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.view(-1) == labels.view(-1)).to(torch.long)
        valid_mask = (labels != self.pad).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = self.compute_contrastive_loss(self.margin, cosine_scores, input_ids, self.pad, prefix_len=0)
        return mle_loss, gen_acc, cl_loss

    def calculate_ppl(self, input_ids, ids_mask, pos_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=ids_mask, position_ids=pos_ids)
        logits = outputs.logits
        mle_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        return math.exp(mle_loss.item())

    def build_mask_matrix(self, seqlen, valid_len_list, prefix_len = 0):
        '''
            prefix_len: the length of prefix that we do not want to compute CL loss for.
            (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
                then the loss padding matrix looks like
                     [0., 1., 1., 1.],
                     [1., 0., 1., 1.],
                     [1., 1., 0., 1.],
                     [1., 1., 1., 0.]
            (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
                then the loss padding matrix looks like
                     [0., 1., 1., 0.],
                     [1., 0., 1., 0.],
                     [1., 1., 0., 0.],
                     [0., 0., 0., 0.]
        '''
        res_list = []
        base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
        base_mask = base_mask.type(torch.FloatTensor)
        bsz = len(valid_len_list)
        for i in range(bsz):
            one_base_mask = base_mask.clone()
            one_valid_len = valid_len_list[i]
            one_base_mask[:,one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
            if prefix_len > 0:
                one_base_mask[:prefix_len, :prefix_len] = 0.
            res_list.append(one_base_mask)
        res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
        #print (res_mask)
        assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
        return res_mask
            
    def compute_contrastive_loss(self, margin, score_matrix, input_ids, pad_token_id, prefix_len=0):
        '''
           margin: predefined margin to push similarity score away
           score_matrix: bsz x seqlen x seqlen
           input_ids: bsz x seqlen
           pad_token_id: indicating which tokens are padding token
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        ### input mask
        input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(pad_token_id), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        valid_len_list = torch.sum(input_mask, dim = -1).tolist()
        loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prefix_len)
        if score_matrix.is_cuda:
            loss_mask = loss_mask.cuda(score_matrix.get_device())
        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
        assert loss_matrix.size() == input_ids.size()
        loss_matrix = loss_matrix * input_mask
        cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
        return cl_loss
    
