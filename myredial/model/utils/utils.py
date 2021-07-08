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


class BertMLEmbedding(nn.Module):

    '''bert model which returns multi layer'''
    
    def __init__(self, model='bert-base-chinese', layer=5):
        super(BertMLEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        self.layer = layer

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(
            ids, 
            attention_mask=attn_mask, 
            output_hidden_states=True
        )
        layers = embds.hidden_states[-self.layer:]
        # L * [B, E]
        layers = [i[:, 0, :] for i in layers]
        return layers

    def load_bert_model(self, state_dict):
        # load the post train checkpoint from BERT-FP (NAACL 2021)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.load_state_dict(new_state_dict)


class BertFullEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BertFullEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask)[0]
        return embds

    def load_bert_model(self, state_dict):
        # load the post train checkpoint from BERT-FP (NAACL 2021)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        # position_ids
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))


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

# ========== Metrics of the BERT-FP ========== #
class Metrics:

    def __init__(self):
        super(Metrics, self).__init__()
        # It depend on positive negative ratio 1:1 or 1:10
        self.segment = 10

    def __process_score_data(self, score_data):
        sessions = []
        one_sess = []
        i = 0
        for score, label in score_data:
            i += 1
            one_sess.append((score, label))
            if i % self.segment == 0:
                one_sess_tmp = np.array(one_sess)
                if one_sess_tmp[:, 1].sum() > 0:
                    # for douban (no positive cases)
                    sessions.append(one_sess)
                one_sess = []
        return sessions

    def __mean_average_precision(self, sort_data):
        #to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1

    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)

    def evaluation_one_session(self, data):
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5

    def evaluate_all_metrics(self, data):
        '''data is a list of double item tuple: [(score, label), ...]'''
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0

        sessions = self.__process_score_data(data)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s)
