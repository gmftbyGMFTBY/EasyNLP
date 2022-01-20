from .InteractionModels import *
from .TraditionalResponseSelectionModels import *
from .WriterRerankModels import *
from .AugmentationModels import *
from .CompareInteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *
from .EvaluationModels import *
from .GenerationModels import *
from .PostTrainModels import *
from .LanguageModels import *
from .SemanticSimilarityModels import *
from .MutualTrainingModels import *

def load_model(args):
    model_type, model_name = args['models'][args['model']]['type'], args['models'][args['model']]['model_name']
    MAP = {
        'Augmentation': AugmentationAgent,
        'Representation': RepresentationAgent,
        'Interaction': InteractionAgent,
        'LatentInteraction': LatentInteractionAgent,
        'Generation': GenerationAgent,
        'CompareInteraction': CompareInteractionAgent,
        'PostTrain': PostTrainAgent,
        'Evaluation': EvaluationAgent,
        'LanguageModel': LanguageModelsAgent,
        'MutualTrainingModel': MutualTrainingAgent,
        'WriterRerank': WriterRerankAgent,
        'SemanticSimilarity': SemanticSimilarityAgent,
        'TraditionalResponseSelection': TraditionalResponseSelectionAgent,
    }
    if model_type in MAP:
        agent_t = MAP[model_type]
    else:
        raise Exception(f'[!] Unknown type {model_type} for {model_name}')

    # vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
    vocab = AutoTokenizer.from_pretrained(args['tokenizer'])
    vocab.add_tokens(['[EOS]'])
    args['vocab_size'] = vocab.vocab_size

    if model_type in ['MutualTrainingModel']:
        model = globals()[model_name](vocab, **args)
        agent = agent_t(vocab, model, args)
    else:
        model = globals()[model_name](**args)
        agent = agent_t(vocab, model, args)
    return agent
