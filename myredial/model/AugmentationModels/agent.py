from model.utils import *
from dataloader.util_func import *

class AugmentationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(AugmentationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
        self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        results = []
        for batch in tqdm(inf_iter):
            ids = batch['ids']
            ids_mask = batch['mask']
            responses = batch['response']
            contexts = batch['context']
            indexes = batch['index']
            res = self.model(ids, ids_mask)
            for c, r, i, re in zip(contexts, responses, indexes, res):
                results.append([c, r, i, re])
        torch.save(results, f'{args["root_dir"]}/data/{args["dataset"]}/train_bert_mask_da.pt')
