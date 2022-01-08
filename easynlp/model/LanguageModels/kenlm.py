from model.utils import *
import kenlm

class KeNLM:

    '''chinese character kenlm n-gram model'''

    def __init__(self, **args):
        super(KeNLM, self).__init__()
        self.args = args
        lm_ckpt_path = self.args['lm_ckpt_path']
        texsmart_path = self.args['texsmart_path']
        self.char_lm = self.args['char_lm']

        self.model = kenlm.LanguageModel(lm_ckpt_path)
        if self.char_lm is False:
            sys.path.append(texsmart_path)
            from tencent_ai_texsmart import NluEngine
            path_head = os.path.split(texsmart_path)[0]
            self.parser = NluEngine(f'{path_head}/data/nlu/kb/', 1)
            print(f'[!] texsmart parser has been loaded')
        print(f'[!] {self.model.order}-gram model has been loaded')

    def cut(self, text):
        if self.char_lm:
            # no need to cut
            return ' '.join(list(text))
        else:
            output = self.parser.parse_text(text)
            sentence = ' '.join([item.str for item in output.words()])
            return sentence

    def predict(self, batch):
        candidates = batch['candidates']
        context = batch['context']
        rest = []
        for candidate in candidates:
            text = self.cut(f'{context} {candidate}')
            ppl = self.model.perplexity(text)
            rest.append(ppl)
        return rest
