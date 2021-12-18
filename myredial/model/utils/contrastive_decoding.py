from .header import *

'''serve for the predict method for GenerationModels: contrastive decoding with beam search'''

def ranking(context_hidden, next_hidden, next_top_k_ids, scoring_criterion, threshold=0.0):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    if scoring_criterion == 'average':
        scores = torch.sum(cosine_matrix, dim = -1)
    else:
        scores, _ = torch.max(cosine_matrix, dim = -1)
    scores = torch.relu(scores - threshold)
    scores = -1 * scores
    _, selected_idx = torch.topk(scores, k = 1)
    selected_idx = selected_idx.unsqueeze(0)
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    return next_id

def ContrastiveDecodingOneStep(model, input_ids, beam_width, scoring_criterion, threshold=0.0):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    output = model(input_ids=input_ids, output_hidden_states=True)
    prev_hidden_states = output.hidden_states[-1]
    logits = output.logits

    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()
    logit_for_next_step = logits[:,-1,:]
    _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
    # compute new hidden 
    expanded_context = [input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim = 0)
    top_k_ids = top_k_ids.view(beam_width, 1)
    next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)

    output = model(input_ids=next_input_ids, output_hidden_states=True)
    new_hidden_states = output.hidden_states[-1]
    context_hidden = new_hidden_states[:,:seqlen,:]
    next_hidden = new_hidden_states[:,seqlen:,:]
    next_id = ranking(context_hidden, next_hidden, top_k_ids, scoring_criterion, threshold)
    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    return next_input_ids
