import torch
from io import StringIO
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import os
import sys
import re
import math
from itertools import chain
import csv
import jieba
from jieba import analyse
import jieba.posseg as pseg
import random
import json
import ijson
import time
import pprint
import hashlib
import logging
from copy import deepcopy
import ipdb
from transformers import BertTokenizer, BertTokenizerFast, XLMRobertaTokenizerFast, AutoModel, AutoTokenizer
import pickle
import argparse
from torch.nn.utils.rnn import pad_sequence
import joblib
from elasticsearch import Elasticsearch, helpers
import faiss
import h5py
import torch.multiprocessing
import linecache
import nanopq

# texsmart
# sys.path.append('/home/johntianlan/sources/texsmart-sdk-0.3.0-m-zh/lib')
# from tencent_ai_texsmart import *

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
