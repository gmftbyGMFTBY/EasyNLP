from model.utils import *


'''seq2seq (bert2bert)'''


class Seq2SeqModel(nn.Module):

    def __init__(self, model, cls):
        super(Seq2SeqModel, self).__init__()
        self.cls_token_id = cls
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model, model)
        # bert-ft pretrained checkpoint
        self.model.encoder.resize_token_embeddings(self.model.encoder.config.vocab_size+1)
        self.model.decoder.resize_token_embeddings(self.model.decoder.config.vocab_size+1)

    def forward(self, cid, cid_mask, rid, rid_mask):
        outputs = self.model(
            input_ids=cid, 
            attention_mask=cid_mask, 
            decoder_input_ids=rid, 
            decoder_attention_mask=rid_mask
        )
        logits = outputs.logits    # [B, S, V]
        hidden = outputs.encoder_last_hidden_state.mean(dim=1)
        return logits, hidden

    def predict(self, cid):
        # Greedy Decoding
        output = self.model.generate(
            cid, 
            do_sample=True,
            decoder_start_token_id=self.cls_token_id,
        )
        return output[0]

    def load_bert_model(self, state_dict):
        # decoder has the different parameters
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.encoder.load_state_dict(new_state_dict)

# serve for the gpt2_contrastive_search model
def compute_valid_token_num(valid_len_list):
    res = 0
    for one_len in valid_len_list:
        res += one_len * (one_len - 1)
    return res

def build_mask_matrix(seqlen, valid_len_list, prefix_len = 0):
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
        
def contrastive_loss(margin, score_matrix, input_ids, pad_token_id, prefix_len=0):
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
    
