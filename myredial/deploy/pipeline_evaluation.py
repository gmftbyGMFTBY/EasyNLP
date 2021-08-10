from header import *
from config import *
from dataloader import *
from inference import Searcher
from es.es_utils import *
from model import Metrics
from .utils import *
from .rerank import *
from .recall import *


class PipelineEvaluationAgent:

    def __init__(self, args):
        self.args = args
        recall_args, rerank_args = args['recall'], args['rerank']
        self.recallagent = RecallAgent(recall_args)
        self.rerankagent = RerankAgent(rerank_args)

        # collection for calculating the metrics
        self.collection = []
        self.metrics = Metrics()

    @timethis
    def work_full_rank_evaluation(self, batch, topk=None, whole_size=0):
        self.metrics.segment = whole_size
        assert len(batch) == 1
        # recall
        topk = topk if topk else self.args['recall']['topk']
        candidates, _ = self.recallagent.work(batch, topk=topk)
        
        # re-packup
        r = [i['text'] for i in candidates[0]]
        rerank_batch = [{'context': batch[0]['str'], 'candidates': r}]

        # rerank
        scores, _ = self.rerankagent.work(rerank_batch)

        # packup
        score, candidate = scores[0], candidates[0]
        idx = np.argmax(score)
        responses = [candidate[idx]['text']]
        ground_truths = batch[0]['ground-truth']
        
        # add the evaluation results
        e_scores, labels = [], []
        gt_overlap = 0
        for s, r in zip(score, candidate):
            r = r['text']
            if r in [ground_truths]:
                labels.append(1)
                gt_overlap += 1
            else:
                labels.append(0)
            e_scores.append(s)
        # add the other samples
        o_scores = [0] * (whole_size - topk)
        o_labels = [1] * (len(ground_truths) - gt_overlap) + [0] * (whole_size - topk + gt_overlap - len(ground_truths))
        labels.extend(o_labels)
        e_scores.extend(o_scores)
        collection = [(s, l) for s, l in zip(e_scores, labels)]
        self.collection.extend(collection)

        scores = self.metrics.evaluate_all_metrics(self.collection)
        for n, s in zip(['MAP', 'MRR', 'P@1', 'R@1', 'R@2', 'R@5'], scores):
            print(f'[!] {n}: {round(s, 4)}')

        return responses
