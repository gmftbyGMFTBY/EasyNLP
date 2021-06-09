import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import ipdb
import unicodedata
import random
from .utils import gelu, LayerNorm
from .transformer_pre import TransformerLayer, Embedding, LearnedPositionalEmbedding


'''Compatible with the api of huggingface transformers'''


def _has_non_chinese_char(s):
    for x in s:
        cp = ord(x)
        if not ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
    return False

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, special tokens, etc.)."""

    def __init__(self, do_lower_case=True, special_tokens=None):
        """Constructs a BasicTokenizer.
        Args:
        do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.special_tokens = special_tokens

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        text = self._tokenize_special_tokens(text)
    
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            if token in self.special_tokens:
                split_tokens.append(token)
            else:
                split_tokens.extend(self._run_split_on_punc(token))
    
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_special_tokens(self, text):
        items = re.split('(\[SEP\]|\[UNK\]|\[MASK\])', text)
        return ' '.join(items)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
          (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials):
        self.num_re = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$") 
        idx2token = specials
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
            
            if self.num_re.match(token) is not None:
                continue 
            if _has_non_chinese_char(token):
                if int(cnt) >= 2*min_occur_cnt:
                    idx2token.append(token)
            else:
                if int(cnt) >= min_occur_cnt:
                    idx2token.append(token)

        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx['[PAD]']
        self._unk_idx = self._token2idx['[UNK]']
        self._num_idx = self._token2idx['[NUM]']
        self._no_chinese_idx = self._token2idx['[NOT_CHINESE]']

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    @property
    def num_idx(self):
        return self._num_idx

    @property
    def no_chinese_idx(self):
        return self._no_chinese_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        if x in self._token2idx:
            return self._token2idx[x]
        return self.unk_idx

class PJBertTokenizer:

    def __init__(self, file_name):
        self.min_occur_cnt = 1000
        self.specials = ['[PAD]', '[UNK]', '[NUM]', '[NOT_CHINESE]', '[CLS]', '[SEP]', '[MASK]'] 
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, special_tokens=self.specials)
        self.vocab = Vocab(file_name, self.min_occur_cnt, self.specials)
        self.size = self.vocab.size
        self.padding_idx = self.vocab.padding_idx
        print(f'[!] load the vocab from {file_name} over, vocab size: {self.size}')

    @classmethod
    def from_pretrained(cls, file_name):
        tokenizer = PJBertTokenizer(file_name)
        return tokenizer

    def convert_tokens_to_ids(token):
        return self.token2idx(token)

    def batch_encode_plus(self, items):
        rest = {'input_ids': [], 'token_type_ids': []}
        types = set([type(item) for item in items])
        assert len(types) == 1, f'[!] Must have the same structure'
        for item in items: 
            if type(item) == str:
                ids = self.basic_tokenizer.tokenize(item)
                ids = ['[CLS]'] + ids + ['[SEP]']
                ids = [self.vocab.token2idx(i) for i in ids]
                tids = [0] * len(ids)
                rest['input_ids'].append(ids)
                rest['token_type_ids'].append(tids)
            elif type(item) == list:
                assert len(item) == 2, '[!] expected 2 utterances, but got: {len(item)}'
                ids_1 = self.basic_tokenizer.tokenize(item[0])
                ids_2 = self.basic_tokenizer.tokenize(item[1])
                ids = ['[CLS]'] + ids_1 + ['[SEP]'] + ids_2 + ['[SEP]']
                ids = [self.vocab.token2idx(i) for i in ids]
                tids = [0] * (len(ids_1) + 2) + [1] * (len(ids_2) + 1)
                rest['input_ids'].append(ids)
                rest['token_type_ids'].append(tids)
            else:
                raise Exception(f'[!] Unknown data type: {type(item)}')
        return rest

    def encode(self, item, add_special_tokens=True):
        if type(item) == str:
            item = self.basic_tokenizer.tokenize(item)
        elif type(item) == list:
            pass
        else:
            raise Exception(f'[!] Cannot handle the {type(item)} for PJBertTokenizer.encode')
        ids = [self.vocab.token2idx(i) for i in item]
        return ids

    def tokenize(self, text):
        return self.basic_tokenizer.tokenize(item)

# ========== BertModel ========== #
class BERTLM(nn.Module):
    def __init__(self, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers, approx):
        super(BERTLM, self).__init__()
        self.vocab = vocab
        self.embed_dim =embed_dim
        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim)
        self.seg_embed = Embedding(2, embed_dim, None)

        self.out_proj_bias = nn.Parameter(torch.Tensor(self.vocab.size))

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.one_more_nxt_snt = nn.Linear(embed_dim, embed_dim) 
        self.nxt_snt_pred = nn.Linear(embed_dim, 1)
        self.dropout = dropout

        if approx == "none":
            self.approx = None
        elif approx == "adaptive":
            self.approx = nn.AdaptiveLogSoftmaxWithLoss(self.embed_dim, self.vocab.size, [10000, 20000, 200000])
        else:
            raise NotImplementedError("%s has not been implemented"%approx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.nxt_snt_pred.bias, 0.)
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.constant_(self.one_more_nxt_snt.bias, 0.)
        nn.init.normal_(self.nxt_snt_pred.weight, std=0.02)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.normal_(self.one_more_nxt_snt.weight, std=0.02)
    
    def forward(self, truth, seg):
        '''truth/seg/mask: [B, S] -> [S, B]'''
        truth = truth.transpose(0, 1)
        seg = seg.transpose(0, 1)

        seq_len, bsz = truth.size()
        x = self.tok_embed(truth) + self.seg_embed(seg) + self.pos_embed(truth)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _ ,_ = layer(x, self_padding_mask=padding_mask)
        return x


class PJBertModel(nn.Module):

    def __init__(self, vocab=None, embed_dim=768, ff_embed_dim=3072, num_heads=12, dropout=0.1, layers=12, approx='none'):
        super(PJBertModel, self).__init__()
        self.model = BERTLM(vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers, approx)

    @classmethod
    def from_pretrained(cls, file_name, **args):
        agent = PJBertModel(**args)
        state_dict = torch.load(file_name)['model']
        agent.model.load_state_dict(state_dict)
        print(f'[!] load checkpoint {file_name} to BERTLM over')
        return agent 

    def forward(self, ids, seg_ids):
        # padding mask are processed in the BERTLM
        embd = self.model(ids, seg_ids)
        # return [B, S, E]
        embd = embd.permute(1, 0, 2)    # [S, B, E] -> [B, S, E]
        return embd


if __name__ == "__main__":
    vocab = PJBertTokenizer.from_pretrained('/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_base/data/vocab.txt')
    rest = vocab.batch_encode_plus([['谢谢你盛情的款待[SEP]不用谢', '真的吗']])
    ids = torch.LongTensor(rest['input_ids'][0])
    tids = torch.LongTensor(rest['token_type_ids'][0])
    model = PJBertModel.from_pretrained('/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_base/ckpt/epoch1_batch_1719999', vocab=vocab, embed_dim=768, ff_embed_dim=3072, num_heads=12, dropout=0.1, layers=12, approx='none')
    model.eval()
    ipdb.set_trace()
    embds = model(ids.unsqueeze(0), tids.unsqueeze(0))
