from .header import *
from .gen_utils import top_k_top_p_filtering

def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, model_prediction_confidence):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    assert cosine_matrix.size() == torch.Size([beam_width, context_len])
    scores, _ = torch.max(cosine_matrix, dim = -1)
    assert scores.size() == torch.Size([beam_width])
    next_top_k_probs = next_top_k_probs.view(-1)
    scores = model_prediction_confidence * next_top_k_probs - (1.0 - model_prediction_confidence) * scores 
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    return next_id

def ContrastiveDecodingOneStep(model, input_ids, beam_width, model_prediction_confidence, top_k, top_p, sampling_probability):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    output = model(input_ids=input_ids, output_hidden_states=True)
    prev_hidden_states = output.hidden_states[-1]
    logits = output.logits

    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()
    p = random.uniform(0, 1)
    if p >= sampling_probability:
        logit_for_next_step = logits[:,-1,:]
        assert logit_for_next_step.size() == torch.Size([1, vocab_size])

        next_probs = F.softmax(logit_for_next_step, dim = -1)
        assert next_probs.size() == logit_for_next_step.size()

        _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
        assert top_k_ids.size() == torch.Size([1, beam_width])
        
        top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

        assert top_k_probs.size() == top_k_ids.size()
        # compute new hidden 
        expanded_context = [input_ids for _ in range(beam_width)]
        expanded_context = torch.cat(expanded_context, dim = 0)
        assert expanded_context.size() == torch.Size([beam_width, seqlen])
        top_k_ids = top_k_ids.view(beam_width, 1)
        next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)
        assert next_input_ids.size() == torch.Size([beam_width, seqlen+1])
        output = model(input_ids=next_input_ids, output_hidden_states=True)
        new_hidden_states = output.hidden_states[-1]
        new_logits = output.logits
        assert new_hidden_states.size() == torch.Size([beam_width, seqlen+1, embed_dim])
        context_hidden = new_hidden_states[:,:seqlen,:]
        assert context_hidden.size() == torch.Size([beam_width, seqlen, embed_dim])
        next_hidden = new_hidden_states[:,seqlen:,:]
        assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])

        next_id = ranking(context_hidden, next_hidden, top_k_ids, top_k_probs, 
            model_prediction_confidence)
    else:
        logit_for_next_step = logits[0,-1,:]
        filtered_logits = top_k_top_p_filtering(logits[0, -1, :], top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        next_id = next_token.view(1, 1)        

    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids
