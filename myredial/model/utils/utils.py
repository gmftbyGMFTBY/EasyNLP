from .header import *


class PositionEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.5, max_len=512):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask)[0]
        embds = embds[:, 0, :]     # [CLS]
        return embds

    def load_bert_model(self, state_dict):
        # load the post train checkpoint from BERT-FP (NAACL 2021)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)

# label smoothing loss
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

'''https://github.com/taesunwhang/BERT-ResSel/blob/master/evaluation.py'''

def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
    total_num_split = len(ground_truth) / eval_candidates_num
    pred_split = np.split(prediction, total_num_split)
    gt_split = np.split(np.array(ground_truth), total_num_split)
    orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
    stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)
    
    rank_by_pred_l = []
    for i, stack_score in enumerate(stack_scores):
        rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
        rank_by_pred = np.stack(rank_by_pred, axis=-1)
        rank_by_pred_l.append(rank_by_pred[0])
    rank_by_pred = np.array(rank_by_pred_l)
    
    pos_index = []
    for sorted_score in rank_by_pred:
        curr_cand = []
        for p_i, score in enumerate(sorted_score):
            if int(score) == 1:
                curr_cand.append(p_i)
        pos_index.append(curr_cand)

    return rank_by_pred, pos_index, stack_scores


def logits_recall_at_k(pos_index, k_list=[1, 2, 5, 10]):
    # 1 dialog, 10 response candidates ground truth 1 or 0
    # prediction_score : [batch_size]
    # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
    # e.g. batch : 100 -> 100/10 = 10

    num_correct = np.zeros([len(pos_index), len(k_list)])
    index_dict = dict()
    for i, p_i in enumerate(pos_index):
        index_dict[i] = p_i

    # case for douban : more than one correct answer case
    for i, p_i in enumerate(pos_index):
        if len(p_i) == 1 and p_i[0] >= 0:
            for j, k in enumerate(k_list):
                if p_i[0] + 1 <= k:
                    num_correct[i][j] += 1
        elif len(p_i) > 1:
            for j, k in enumerate(k_list):
                all_recall_at_k = []
                for cand_p_i in p_i:
                    if cand_p_i + 1 <= k:
                        all_recall_at_k.append(1)
                    else:
                        all_recall_at_k.append(0)
                num_correct[i][j] += np.mean(all_recall_at_k)
                # num_correct[i][j] += np.max(all_recall_at_k)

    return np.sum(num_correct, axis=0)

def logits_mrr(pos_index):
    mrr = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0 and p_i[0] >= 0:
            mrr.append(1 / (p_i[0] + 1))
        elif len(p_i) == 0:
            mrr.append(0)  # no answer

    return np.sum(mrr)

def precision_at_one(rank_by_pred):
    num_correct = [0] * rank_by_pred.shape[0]
    for i, sorted_score in enumerate(rank_by_pred):
        for p_i, score in enumerate(sorted_score):
            if p_i == 0 and int(score) == 1:
                num_correct[i] = 1
                break

    return np.sum(num_correct)

def mean_average_precision(pos_index):
    map = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0:
            all_precision = []
            for j, cand_p_i in enumerate(p_i):
                all_precision.append((j + 1) / (cand_p_i + 1))
            curr_map = np.mean(all_precision)
            map.append(curr_map)
        elif len(p_i) == 0:
            map.append(0)  # no answer

    return np.sum(map)
