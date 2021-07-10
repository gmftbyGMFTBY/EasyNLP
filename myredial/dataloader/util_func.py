import torch
import ipdb
import random

def truncate_pair(cids, rids, max_length):
    # change the cids and rids in place
    max_length -= 3    # [CLS], [SEP], [SEP]
    while True:
        l = len(cids) + len(rids)
        if l <= max_length:
            break
        if len(cids) > 2 * len(rids):
            cids.pop(0)
        else:
            rids.pop()


def truncate_pair_two_candidates(cids, rids1, rids2, max_length):
    max_length -= 4    # [CLS] ctx [SEP] rids1 [SEP] rids2 [SEP]
    while True:
        l = len(cids) + len(rids1) + len(rids2)
        if l <= max_length:
            break
        if len(cids) > len(rids1) + len(rids2):
            cids.pop(0)
        elif len(rids1) > len(rids2):
            rids1.pop()
        else:
            rids2.pop()


def generate_mask(ids):
    '''generate the mask matrix of the ids'''
    attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
    attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
    attn_mask = torch.zeros_like(ids)
    attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
    return attn_mask


def to_cuda(*args):
    '''map the tensor on cuda device'''
    if not torch.cuda.is_available():
        return args
    tensor = []
    for i in args:
        i = i.cuda()
        tensor.append(i)
    return tensor


def mask_sentence(
        ids, min_mask_num, max_mask_num, masked_lm_prob, 
        special_tokens=[], mask=-1, vocab_size=21128,
    ):
    '''change the ids, and return the mask_label'''
    num_valid = len([i for i in ids if i not in special_tokens])
    num_mask = max(
        min_mask_num,
        min(
            int(masked_lm_prob * num_valid),
            max_mask_num,
        )
    )

    mask_pos = [idx for idx, i in enumerate(ids) if i not in special_tokens]
    mask_idx = random.sample(mask_pos, num_mask)
    mask_label = []
    for idx, i in enumerate(ids):
        if idx in mask_idx:
            ratio = random.random()
            if ratio < 0.8:
                ids[idx] = mask
            elif ratio < 0.9:
                # random change
                ids[idx] = random.choice(list(range(vocab_size)))
            mask_label.append(i)
        else:
            mask_label.append(-1)
    return mask_label

# ========== dual-bert ========== #
def length_limit(ids, max_len):
    # also return the speaker embeddings
    if len(ids) > max_len:
        ids = [ids[0]] + ids[-(max_len-1):]
    return ids

def length_limit_res(ids, max_len, sep=0):
    # cut tail
    if len(ids) > max_len:
        ids = ids[:max_len-1] + [sep]
    return ids
