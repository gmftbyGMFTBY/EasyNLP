from .InteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *
from .GenerationModels import *
from .PostTrainModels import *

def load_model(args):
    model_type, model_name = args['models'][args['model']]['type'], args['models'][args['model']]['model_name']
    MAP = {
        'Representation': RepresentationAgent,
        'Interaction': InteractionAgent,
        'LatentInteraction': LatentInteractionAgent,
        'Generation': GenerationAgent,
    }
    if model_type in MAP:
        agent_t = MAP[model_type]
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
