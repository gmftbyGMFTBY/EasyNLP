from .InteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *

def load_model(args):
    MAP = {
        'bert-ft': BERTFTAgent,
        'sa-bert': SABERTFTAgent,
        'dual-bert-gray': BERTDualWriterEncoderAgent,
        'dual-bert': BERTDualEncoderAgent,
        'poly-encoder': BERTPolyEncoderAgent,
        'dual-bert-hierarchical-trs': BERTDualHierarchicalTrsEncoderAgent,
    }
    if args['model'] in MAP:
        return MAP[args['model']](args)
    else:
        raise Exception(f'[!] cannot find the model: {args["model"]}')
