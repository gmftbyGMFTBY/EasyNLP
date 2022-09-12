from evaluate import load
from model.utils import *

class BERTScoreModel(nn.Module):

    def __init__(self, **args):
        super(BERTScoreModel, self).__init__()
        self.scorer = load('bertscore', device=f'cuda:0')

    def predict(self, batch):
        results = self.scorer.compute(predictions=batch['response'], references=batch['reference'], lang='zh')
        return results['f1']
