from .InteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *

def load_model(args):
    model_type, model_name = args['models'][args['model']]['type'], args['models'][args['model']]['model_name']
    if model_type == 'Representation':
        agent_t = RepresentationAgent
    elif model_type == 'Interaction':
        agent_t = InteractionAgent
    elif model_type == 'LatentInteraction':
        agent_t = LatentInteractionAgent
    else:
        raise Exception(f'[!] Unknown type {model_type} for {model_name}')

    if 'pj-' in args['model']:
        # load pj bert model
        vocab = PJBertTokenizer.from_pretrained(args['tokenizer'])
        args['vocab_size'] = vocab.size
        args['padding_idx'] = vocab.padding_idx
        model = globals()[model_name](**args)
    else:
        vocab = BertTokenizer.from_pretrained(args['tokenizer'])
        model = globals()[model_name](**args)
    agent = agent_t(vocab, model, args)
    return agent
