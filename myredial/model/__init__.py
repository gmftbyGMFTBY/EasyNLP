from .InteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *
from .GenerationModels import *

def load_model(args):
    model_type, model_name = args['models'][args['model']]['type'], args['models'][args['model']]['model_name']
    if model_type == 'Representation':
        agent_t = RepresentationAgent
    elif model_type == 'Interaction':
        agent_t = InteractionAgent
    elif model_type == 'LatentInteraction':
        agent_t = LatentInteractionAgent
    elif model_type == 'Generation':
        agent_t = GenerationAgent
    else:
        raise Exception(f'[!] Unknown type {model_type} for {model_name}')

    vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
    args['vocab_size'] = vocab.vocab_size

    if model_type == 'Generation':
        model = globals()[model_name](vocab, **args)
    else:
        model = globals()[model_name](**args)
    agent = agent_t(vocab, model, args)
    return agent
