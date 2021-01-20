from .header import *
from .base import *
from .utils import *

'''Generative Augmentation Retrieval Dual-Bert model:
1. Context encoder (UniLM Mask):
    1.1 BERT model select [CLS] token (context fully self-attention)
    1.2 BERT model 

2. Candidate encoder:
BERT model select the [CLS] token
'''