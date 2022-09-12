from evaluate import load
from model.utils import *

class BLEUModel(nn.Module):

    def __init__(self, **args):
        super(BLEUModel, self).__init__()
        self.scorer = load('bleu')

    def predict(self, batch):
        references = [[' '.join(list(i))] for i in batch['reference']]
        response = [' '.join(list(i)) for i in batch['response']]
        results = self.scorer.compute(predictions=response, references=references)
        return [results['bleu']]
