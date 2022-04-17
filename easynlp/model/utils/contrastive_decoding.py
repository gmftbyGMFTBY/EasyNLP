from .header import *
from .gen_utils import *

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

def ContrastiveDecodingOneStep(model, input_ids, beam_width, model_prediction_confidence, unk_id):
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
    logit_for_next_step[:, unk_id] = -np.inf
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
    sep_idx, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    step,
    is_sampling
    ):
    # input_ids: [B, S]
    if step == 0:
        output = model(
            input_ids=ids, 
            attention_mask=ids_mask,
            position_ids=ids_pos,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    if is_sampling is False:
        # next_probs = F.softmax(logit_for_next_step, dim=-1)
        next_probs = F.softmax(logit_for_next_step, dim=-1)
        _, top_k_ids = torch.topk(next_probs, dim=-1, k=beam_width)    # [B, K]
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
        # next stage: move forward one step to rerank the tokens by the motivation of the contrastive search
        past_key_values = enlarge_past_key_values(past_key_values, beam_width)
        ids_pos_new = ids_pos.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz*beam_width, -1)[:, -1].unsqueeze(dim=-1) + 1
        output = model(
            input_ids=top_k_ids.view(-1, 1), 
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            position_ids=ids_pos_new,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]    # [B*K, V]
        next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
        context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

        selected_idx = ranking_batch(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            model_prediction_confidence,
            beam_width,
        )     # [B]
        # prepare for the next step
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
        next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
        past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    else:
        filtered_logits = top_k_top_p_filtering_batch(logit_for_next_step, top_k=top_k, top_p=top_p)
        next_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        # also move forward one step: inactivate the enlarge_past_key_values
        output = model(
            input_ids=next_id, 
            attention_mask=torch.ones_like(next_id),
            position_ids=ids_pos[:, -1].unsqueeze(dim=-1) + 1,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]    # [B, V]
        next_hidden = output.hidden_states[-1]    # [B, 1, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden], dim=1)    # [B, S, E]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits 

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

# beam search version contrastive search for diverse generations
def ContrastiveDecodingOneStepBeamSearch(
    model, 
    ids, 
    beam_width, 
    model_prediction_confidence, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    step,
    contrastive_generation_num,
    queue,
    queue_scores,
    limited_size,
    delta,
    ):
    if step == 0:
        output = model(
            input_ids=ids, 
            use_cache=True,
            output_hidden_states=True
        )
        # past_key_values = output.past_key_values
        logit_for_next_step = output.logits[:, -1, :]    # [1, V]

        # move one step further to generate the queue with the beam size
        next_id = logit_for_next_step.topk(dim=-1, k=beam_width)[1].t()     # [B, 1]
        queue_scores = F.softmax(logit_for_next_step, dim=-1).squeeze(0)[next_id.t().squeeze(0)]    # [B]
        ids = torch.cat([ids.expand(beam_width, -1), next_id], dim=-1)    # [B, S+1]
        output = model(
            input_ids=ids, 
            use_cache=True, 
            output_hidden_states=True, 
            # past_key_values=past_key_values
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]

        # init the queue
        queue = [item.tolist() for item in ids]

    _, seqlen, embed_dim = last_hidden_states.size()

    # next stage: move forward one step to rerank the tokens by the motivation of the contrastive search
    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(next_probs, dim=-1, k=beam_width)    # [B, B]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, B]
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1), 
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*B, V]
    next_hidden = output.hidden_states[-1]    # [B*B, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(len(next_hidden), seqlen, embed_dim)    # [B*B, S, E]

    selected_idx = ranking_beam_search(
        context_hidden, 
        next_hidden, 
        top_k_probs,
        model_prediction_confidence,
        beam_width,
        queue_scores,
        limited_size,
    )    # [B]

    # get the father node in the search graph
    new_queue = []
    top_k_ids = top_k_ids.view(-1, 1)
    for node in selected_idx.tolist():
        father_index = node // beam_width 
        father = deepcopy(queue[father_index])
        current_node = top_k_ids[node].item()
        rest = father + [current_node]
        new_queue.append(rest)

    # prepare for the next step
    next_id = top_k_ids[selected_idx, :]    # [B, 1]
    next_hidden = next_hidden[selected_idx, :]    # [B, 1, E]
    last_hidden_states = torch.cat([context_hidden[selected_idx, :, :], next_hidden], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values_beam(past_key_values, selected_idx)
    logits = logits[selected_idx, :]    # [B, V]
    return next_id, past_key_values, last_hidden_states, logits, new_queue, top_k_probs 

def ranking_beam_search(context_hidden, next_hidden, next_top_k_probs, model_prediction_confidence, select_num, queue_scores, limited_size):
    '''
        context_hidden: beam*beam x seqlen x embed_dim
        next_hidden: beam*beam x 1 x embed_dim
        next_top_k_probs: beam x beam
        queue_scores: beam -> beam x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*B, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*B]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*B]
    scores = model_prediction_confidence * next_top_k_probs - (1.0 - model_prediction_confidence) * scores    # [B*B] 

    # limited the search results
    scores = torch.stack(torch.split(scores, select_num))    # [B, B]
    sub_scores, sub_scores_idx = scores.topk(dim=-1, k=limited_size)    # [B, L]
    delta = []
    for i in range(select_num):
        delta.extend([i * select_num] * limited_size)
    delta = torch.LongTensor(delta).cuda()
    sub_scores, sub_scores_idx = sub_scores.view(-1), sub_scores_idx.view(-1)    # [B*L]
    sub_scores_idx += delta
    _, sub_sub_scores_idx = sub_scores.topk(k=select_num)
    sub_scores_idx = sub_scores_idx[sub_sub_scores_idx]
    return sub_scores_idx

    # consider the past scores
    # _, selected_idx = scores.topk(k=select_num)   # [B]
    # return selected_idx

def select_past_key_values_beam(past_key_values, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            item = item[selected_idx, :, :, :]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


# ========== ========== #
def TokenRerankDecodingOneStep(model, reranker, input_ids, beam_width, model_prediction_confidence, top_k, top_p, sampling_probability, sep_idx, sep_smooth_length):
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
    new_hidden_score = F.softmax(reranker(new_hidden_states), dim=-1)[:, -1, 1]    # [beam_width]
    new_hidden_score = model_prediction_confidence * top_k_probs.squeeze(0) + (1-model_prediction_confidence) * new_hidden_score
    best_index = new_hidden_score.max(dim=-1)[1]
    next_id = top_k_ids[best_index, :]     # [1]
    next_input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=-1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids

# ========== SimRAG ========== #
def ContrastiveDecodingOneStepSimRAGBatch(
    model, 
    rag_hidden,    # [B, E]
    ids, 
    ids_mask,
    ids_pos,
    beam_width, 
    model_prediction_confidence, 
    top_k, 
    top_p, 
    sep_idx, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    step,
    is_sampling,
    beta,
    beta_scale
    ):
    # input_ids: [B, S]
    if step == 0:
        output = model(
            input_ids=ids, 
            attention_mask=ids_mask,
            position_ids=ids_pos,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    if is_sampling is False:
        next_probs = F.softmax(logit_for_next_step, dim=-1)
        _, top_k_ids = torch.topk(next_probs, dim=-1, k=beam_width)    # [B, K]
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
        # next stage: move forward one step to rerank the tokens by the motivation of the contrastive search
        past_key_values = enlarge_past_key_values(past_key_values, beam_width)
        ids_pos_new = ids_pos.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz*beam_width, -1)[:, -1].unsqueeze(dim=-1) + 1
        output = model(
            input_ids=top_k_ids.view(-1, 1), 
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            position_ids=ids_pos_new,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]    # [B*K, V]
        next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
        context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

        selected_idx = ranking_simrag_batch(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            model_prediction_confidence,
            beam_width,
            rag_hidden,    # [B_rag, E]
            beta,
            beta_scale
        )     # [B]
        # prepare for the next step
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
        next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
        past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    else:
        filtered_logits = top_k_top_p_filtering_batch(logit_for_next_step, top_k=top_k, top_p=top_p)
        next_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        # also move forward one step: inactivate the enlarge_past_key_values
        output = model(
            input_ids=next_id, 
            attention_mask=torch.ones_like(next_id),
            position_ids=ids_pos[:, -1].unsqueeze(dim=-1) + 1,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]    # [B, V]
        next_hidden = output.hidden_states[-1]    # [B, 1, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden], dim=1)    # [B, S, E]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits 

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def ranking_simrag_batch(context_hidden, next_hidden, next_top_k_probs, model_prediction_confidence, beam_width, rag_hidden, beta, beta_scale):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
        rag_hidden: bsz_rag x embed_dim
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    norm_rag_hidden = rag_hidden / rag_hidden.norm(dim=1, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]

    # rag scores matrix
    rag_scores = torch.matmul(norm_next_hidden.squeeze(1), norm_rag_hidden.t()).max(dim=-1)[0] * beta_scale    # [B*K]
    # rag_scores = torch.matmul(norm_next_hidden.squeeze(1), norm_rag_hidden.t()).mean(dim=-1) * beta_scale    # [B*K]

    scores = model_prediction_confidence * next_top_k_probs - (1.0 - model_prediction_confidence - beta) * scores + beta * rag_scores
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx


# ========== plug and play simrag contrastive seach one step
def PlugAndPlayRAGContrastiveDecodingOneStep(
    model, model_tokenizer, scorer, scorer_tokenizer, input_ids, beam_width, alpha, beta, rag_sentence, prefix_length
):
    output = model(input_ids, output_hidden_states=True)
    prev_hidden_states, logits = output['hidden_states'][-1], output['logits']
    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()

    logit_for_next_step = logits[:,-1,:]    # [1, V]
    next_probs = F.softmax(logit_for_next_step, dim = -1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [1, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [1, K]

    # compute new hidden 
    expanded_context = [input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim=0)    # [K, S]
    top_k_ids = top_k_ids.view(beam_width, 1)    # [K, 1]
    next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)    # [K, S+1]

    # compute simcse score
    batch_text_list = []
    for one_input_id in next_input_ids:
        one_text = model_tokenizer.decode(one_input_id[prefix_length:])
        batch_text_list.append(one_text)
    simcse_score = scorer.predict_score(batch_text_list, rag_sentence)    # [1, K]

    output = model(next_input_ids, output_hidden_states=True)
    new_hidden_states, next_logits = output['hidden_states'][-1], output['logits']
    context_hidden = new_hidden_states[:,:seqlen,:]    # [K, S, E]
    next_hidden = new_hidden_states[:,seqlen:,:]    # [K, 1, E]

    next_id = plug_and_play_ranking(context_hidden, next_hidden, top_k_ids, top_k_probs, alpha, beta, simcse_score)       
    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    return next_input_ids

def plug_and_play_ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, 
    alpha, beta, batch_class_score):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
        batch_class_score: beam_width x 1
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
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores + beta * batch_class_score.view([beam_width])
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    return next_id


# ===== contrastive search for copy generation ===== #
def ContrastiveDecodingOneStepForCopyGeneration(model, input_ids, beam_width, model_prediction_confidence, sep_idx, logits, prev_hidden_states):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    # output = model(input_ids=input_ids, output_hidden_states=True)
    # prev_hidden_states = output.hidden_states[-1]
    # logits = output.logits
    # hidden_state_for_copy = prev_hidden_states[:, -1, :]    # [B, E]

    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()
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

    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids
