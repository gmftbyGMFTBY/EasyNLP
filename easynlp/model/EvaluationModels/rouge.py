from evaluate import load
from model.utils import *

class ROUGEModel(nn.Module):

    def __init__(self, **args):
        super(ROUGEModel, self).__init__()
        self.scorer = load('rouge')

    def predict(self, batch):
        references = [' '.join(list(i)) for i in batch['reference']]
        response = [' '.join(list(i)) for i in batch['response']]
        results = self.scorer.compute(predictions=response, references=references)
        return [results['rougeL']]
