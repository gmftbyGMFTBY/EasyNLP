from model.utils import *
from model.InteractionModels import *
from model.RepresentationModels import *
from config import *

class ScorerAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(ScorerAgent, self).__init__()
        # load the interactionagent and representationagent
        bert_ft_args = deepcopy(args)
        bert_ft_args['model'] = 'bert-ft'
        config = load_config(bert_ft_args)
        bert_ft_args.update(config)
        vocab = AutoTokenizer.from_pretrained(bert_ft_args['tokenizer']) 
        vocab.add_tokens(['[EOS]'])
        bert_ft_args['vocab_size'] = vocab.vocab_size
        model = BERTRetrieval(**bert_ft_args) 
        self.interactionagent = InteractionAgent(vocab, model, bert_ft_args)
        pretrained_model = bert_ft_args['pretrained_model'].replace('/', "_")
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/bert-ft/best_{pretrained_model}_{args["version"]}.pt'
        self.interactionagent.load_model(save_path)

        # load the representatnionagent
        dr_bert_args = deepcopy(args)
        dr_bert_args['model'] = 'dual-bert'
        config = load_config(dr_bert_args)
        dr_bert_args.update(config)
        vocab = AutoTokenizer.from_pretrained(dr_bert_args['tokenizer']) 
        vocab.add_tokens(['[EOS]'])
        dr_bert_args['vocab_size'] = vocab.vocab_size
        model = BERTDualEncoder(**dr_bert_args) 
        self.representationagent = RepresentationAgent(vocab, model, dr_bert_args)
        pretrained_model = dr_bert_args['pretrained_model'].replace('/', "_")
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/dual-bert/best_{pretrained_model}_{args["version"]}.pt'
        self.representationagent.load_model(save_path)

    def load_model(self, path):
        pass
    
    @torch.no_grad()
    def inference_clean(self, inf_iter, inf_data, size=100000):
        pbar = tqdm(inf_iter)
        results, writers = [], []
        for batch in pbar:
            if batch['ids'] is None:
                break
            raws = batch['raw']
            bert_ft_scores = self.interactionagent.model(batch)    # [B, 2]
            bert_ft_scores = F.softmax(bert_ft_scores, dim=-1)[:, 1].tolist()    # [B]
            dr_bert_scores = self.representationagent.model.module.score(batch).tolist()
            for raw, s1, s2 in zip(raws, bert_ft_scores, dr_bert_scores):
                raw['bert_ft_score'] = round(s1, 4)
                raw['dr_bert_score'] = round(s2, 4)
            results.extend(raws)
            writers.extend(batch['writers'])
            if len(results) >= size:
                for rest, fw in zip(results, writers):
                    string = json.dumps(rest, ensure_ascii=False) + '\n'
                    fw.write(string)
                results, writers = [], []
        if len(results) > 0:
            for rest, fw in zip(results, writers):
                string = json.dumps(rest, ensure_ascii=False) + '\n'
                fw.write(string)
