from evaluate import load
from model.utils import *

class BLEURTModel(nn.Module):

    def __init__(self, **args):
        super(BLEURTModel, self).__init__()
        self.scorer = load('bleurt', device='cuda:0', model_type='metric')

    def predict(self, batch):
        results = self.scorer.compute(predictions=batch['response'], references=batch['reference'])
        return results['scores']
