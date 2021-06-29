from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher
from es.es_utils import *
from .utils import *
from .rerank import *
from .recall import *


class PipelineAgent:

    def __init__(self, recall_args, rerank_args):
        self.recallagent = RecallAgent(recall_args)
        self.rerankagent = RerankAgent(rerank_args)

    @timethis
    def work(self, batch):
        # recall
        candidates, _ = self.recallagent.work(batch, topk=None)
        
        # re-packup
        contexts = [i['str'] for i in batch]
        rerank_batch = []
        for c, r in zip(contexts, candidates):
            rerank_batch.append({'context': c, 'candidates': r})

        # rerank
        scores, _ = self.rerankagent.work(rerank_batch)

        # packup
        responses = []
        for score, candidate in zip(scores, candidates):
            idx = np.argmax(score)
            responses.append(candidate[idx])
        return responses


