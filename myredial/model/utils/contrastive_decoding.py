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

def ContrastiveDecodingOneStep(model, input_ids, beam_width, model_prediction_confidence, top_k, top_p, sampling_probability, sep_idx, sep_smooth_length):
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
        # ignore sep
        logit_for_next_step[:, sep_idx] *= sep_smooth_length
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

# ========== batch version ========= #
def ranking_batch(context_hidden, next_hidden, next_top_k_probs, model_prediction_confidence, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = model_prediction_confidence * next_top_k_probs - (1.0 - model_prediction_confidence) * scores 
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def ContrastiveDecodingOneStepBatch(
    model, 
    ids, 
    ids_mask,
    ids_pos,
    beam_width, 
    model_prediction_confidence, 
    top_k, 
    top_p, 
    sampling_probability, 
    sep_idx, 
    sep_smooth_length,
    past_key_values,
    last_hidden_states,
    vocab,
    ):
    # input_ids: [B, S] -> [B*K, S]
    _, seqlen = ids.size()
    ids = ids.unsqueeze(1).expand(-1, beam_width, -1).reshape(-1, seqlen)
    ids_pos = ids_pos.unsqueeze(1).expand(-1, beam_width, -1).reshape(-1, seqlen)
    ids_mask = ids_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(-1, seqlen)

    output = model(
        input_ids=ids, 
        attention_mask=ids_mask,
        position_ids=ids_pos,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True
    )
    past_key_values = output.past_key_values
    current_hidden_states = output.hidden_states[-1]    # [B*K, S, E]
    if last_hidden_states is not None:
        last_hidden_states = torch.cat([last_hidden_states, current_hidden_states], dim=1)
    else:
        last_hidden_states = current_hidden_states
    bsz, seqlen, embed_dim = last_hidden_states.size()
    logits = output.logits    # [B*K, S, V]
    _, _, vocab_size = logits.size()
    p = random.uniform(0, 1)
    if p >= sampling_probability:
        logit_for_next_step = logits[range(0, bsz, beam_width), -1, :]    # [B, V]
        logit_for_next_step[:, sep_idx] *= sep_smooth_length
        next_probs = F.softmax(logit_for_next_step, dim=-1)
        _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
        # compute new hidden
        output = model(
            input_ids=top_k_ids.view(-1, 1), 
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            position_ids=1+ids_pos[:, -1].unsqueeze(dim=-1),
            past_key_values=past_key_values,
            output_hidden_states=True
        )
        next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
        context_hidden = last_hidden_states.clone()    # [B*K, S, E]

        selected_idx = ranking_batch(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            model_prediction_confidence,
            beam_width,
        )     # [B]
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    else:
        logit_for_next_step = logits[0, -1, :]
        filtered_logits = top_k_top_p_filtering(logits[0, -1, :], top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_id = torch.multinomial(probabilities, 1)
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states 
