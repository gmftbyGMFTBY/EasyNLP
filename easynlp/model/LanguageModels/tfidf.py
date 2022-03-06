from model.utils import *
from dataloader import *
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import matutils

class TFIDFModel:

    def __init__(self, **args):
        super(TFIDFModel, self).__init__()
        self.data = []
        self.dct_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/tfidf/tfidf.dict'
        self.model_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/tfidf/tfidf.model'
        if os.path.exists(self.dct_path) and os.path.exists(self.model_path):
            self.dct = Dictionary.load(self.dct_path)
            self.model = TfidfModel.load(self.model_path)
            self.vocab_size = len(self.dct.token2id)
            print(f'[!] load the model and the dictionary from:\n - {self.dct_path}\n - {self.model_path}')
            return

        # preprocessing
        data_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
        data = read_text_data_utterances(data_path)
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            utterances = [list(jieba.cut(utterance)) for utterance in utterances]
            self.data.extend(utterances)
        self.dct = Dictionary(self.data)
        corpus = [self.dct.doc2bow(doc) for doc in self.data]
        self.model = TfidfModel(corpus)
        self.dct.save(self.dct_path)
        self.model.save(self.model_path)
        print(f'[!] save the model and the dictionary into:\n - {self.dct_path}\n - {self.model_path}')

    def predict(self, batch):
        texts = [self.dct.doc2bow(list(jieba.cut(doc))) for doc in batch]
        inpt = [self.model[text] for text in texts]
        vecs = matutils.corpus2dense(inpt, self.vocab_size)
        return torch.from_numpy(vecs.transpose()).cuda()
