from .header import *

'''
Base Agent
'''

class RetrievalBaseAgent:

    def __init__(self):
        # NOTE: for torch.cuda.amp
        self.scaler = GradScaler()

    def show_parameters(self, args):
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def load_model(self, path):
        '''
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel(self.model) are used to load the model
        '''
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        try:
            self.model.module.load_state_dict(state_dict)
        except:
            self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError
