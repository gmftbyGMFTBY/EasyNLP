from model.utils import *
from config import load_model_config
from .dual_bert_triplet_mutual import *
from .sa_bert_mutual import *

class TransBiEncoder(nn.Module):

    '''Cross-encoder re-label the context-response pairs for improving bi-encoder'''

    def __init__(self, vocab, **args):
        super(TransBiEncoder, self).__init__()
        self.args = args
        self.vocab = vocab
        self.sep, self.cls, self.pad, self.eos = self.vocab.convert_tokens_to_ids([
            '[SEP]', '[CLS]', '[PAD]', '[EOS]'
        ])

        # load bi-encoder model
        config = deepcopy(self.args)
        config.update(load_model_config(self.args['bi_encoder'], self.args['mode']))
        self.bi_encoder_model = BERTDualTripletMarginMutualEncoder(**config)
        # load cross-encoder model
        config = deepcopy(self.args)
        config.update(load_model_config(self.args['cross_encoder'], self.args['mode']))
        self.cross_encoder_model = SABERTMutualRetrieval(**config)
        self.training_model = args['training_model']

    @torch.no_grad()
    def cross_encoder_predict(self, batch):
        self.cross_encoder_model.eval()
        return self.cross_encoder_model.predict(batch)

    @torch.no_grad()
    def bi_encoder_predict(self, batch):
        self.bi_encoder_model.eval()
        return self.bi_encoder_model.predict(batch)

    @torch.no_grad()
    def obtain_training_batch_for_bi_encoder(self, batch):
        '''cross-encoder for soft labeling'''
        context, response, candidates, easy = deepcopy(batch['context']), deepcopy(batch['response']), deepcopy(batch['candidates']), deepcopy(batch['easy'])
        topk = self.args['gray_cand_num']
        candidates = list(chain(*candidates))
        easy = list(chain(*easy))
        context = list(chain(*[[i]*topk for i in context]))
        response = list(chain(*[[i]*topk for i in response]))
        ids = self.text_to_tensor_bi_encoder_ctx(context)
        rids = self.text_to_tensor_bi_encoder_res(response)
        cids = self.text_to_tensor_bi_encoder_res(candidates)
        eids = self.text_to_tensor_bi_encoder_res(easy)

        # for cross-encoder labeling
        ids_, sids_, tids_ = self.text_to_tensor_cross_encoder(context, candidates)
        ids_ = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
        sids_ = pad_sequence(sids_, batch_first=True, padding_value=self.pad)
        tids_ = pad_sequence(tids_, batch_first=True, padding_value=self.pad)
        ids_mask_ = generate_mask(ids_)
        ids_, sids_, tids_, ids_mask_ = to_cuda(ids_, sids_, tids_, ids_mask_)
        soft_label = self.cross_encoder_predict({
            'ids': ids_,
            'sids': sids_,
            'tids': tids_,
            'mask': ids_mask_,
        })    # [B*K]
        # filter out the false negative samples
        ids_n, rids_n, cids_n, eids_n = [], [], [], []
        for idx, l in enumerate(soft_label.tolist()):
            if l < self.args['detect_margin']:
                ids_n.append(ids[idx])
                rids_n.append(rids[idx])
                cids_n.append(cids[idx])
        if len(ids_n) == 0:
            # no samples for training
            batches = None
        else:
            # sample easy negative samples for training
            eids_n = random.sample(eids, len(ids_n))
            batches = {
                'ids': ids_n,
                'rids': rids_n,
                'hrids': cids_n,
                'erids': eids_n,
            }
        # for dot product contrastive loss optimization
        context, response = deepcopy(batch['context']), deepcopy(batch['response'])
        ids = self.text_to_tensor_bi_encoder_ctx(context)
        rids = self.text_to_tensor_bi_encoder_res(response)
        ids =pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids =pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask, rids_mask = generate_mask(ids), generate_mask(rids)
        ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
        dp_batch = {
            'ids': ids,
            'rids': rids,
            'ids_mask': ids_mask,
            'rids_mask': rids_mask,
        }
        return batches, dp_batch

    def forward(self, batch):
        batches, dp_batch = self.obtain_training_batch_for_bi_encoder(batch)
        loss = 0
        if batches is not None:
            for i in range(0, len(batches['ids']), self.args['inner_bsz']):
                sub_ids = pad_sequence(
                    batches['ids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_rids = pad_sequence(
                    batches['rids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_hrids = pad_sequence(
                    batches['hrids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_erids = pad_sequence(
                    batches['erids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_ids_mask = generate_mask(sub_ids)
                sub_hrids_mask = generate_mask(sub_hrids)
                sub_rids_mask = generate_mask(sub_rids)
                sub_erids_mask = generate_mask(sub_erids)
                sub_ids, sub_rids, sub_hrids, sub_erids, sub_ids_mask, sub_rids_mask, sub_hrids_mask, sub_erids_mask = to_cuda(sub_ids, sub_rids, sub_hrids, sub_erids, sub_ids_mask, sub_rids_mask, sub_hrids_mask, sub_erids_mask)
                batch = {
                    'ids': sub_ids,
                    'ids_mask': sub_ids_mask,
                    'rids': sub_rids,
                    'rids_mask': sub_rids_mask,
                    'hrids': sub_hrids,
                    'hrids_mask': sub_hrids_mask,
                    'erids': sub_erids,
                    'erids_mask': sub_erids_mask,
                }
                loss += self.bi_encoder_model(batch)
        loss += self.bi_encoder_model(dp_batch, loss_type='contrastiveloss')
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # clip_grad_norm_(self.parameters(), self.args['grad_clip'])
        # scaler.step(optimizer)
        # scaler.update()
        return loss

    @torch.no_grad()
    def predict(self, batch, test_model_name='bi-encoder'):
        context, responses = batch['context'], batch['responses']
        context = [context] * 10
        ids = self.text_to_tensor_bi_encoder_ctx(context)
        rids = self.text_to_tensor_bi_encoder_res(responses)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        rids_mask = generate_mask(rids)
        ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
        return self.bi_encoder_predict({
            'ids': ids,
            'ids_mask': ids_mask,
            'rids': rids,
            'rids_mask': rids_mask,
        })

    def text_to_tensor_bi_encoder_ctx(self, ctx):
        ids = []
        for c in ctx:
            items = self.vocab.batch_encode_plus(c, add_special_tokens=False)['input_ids']
            ids_ = []
            for u in items:
                ids_.extend(u + [self.sep])
            ids_.pop()
            ids_ = [self.cls] + ids_[-self.args['max_len']+2:] + [self.sep]
            ids.append(torch.LongTensor(ids_))
        return ids

    def text_to_tensor_bi_encoder_res(self, res):
        rids = []
        for r in res:
            items = self.vocab.encode(r, add_special_tokens=False)
            rids.append(
                torch.LongTensor(
                    [self.cls] + items[:self.args['res_max_len']-2] + [self.sep]
                )
            )
        return rids

    def text_to_tensor_cross_encoder(self, ctx, res):
        ids, tids, sids = [], [], []
        for c, r in zip(ctx, res):
            ids_, tids_, sids_ = self.sa_cross_encoder_annotate(c + [r])
            ids.append(torch.LongTensor(ids_))
            tids.append(torch.LongTensor(tids_))
            sids.append(torch.LongTensor(sids_))
        return ids, tids, sids

    def sa_cross_encoder_annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, sids, tcache, scache = [], [], [], 0, 0
        for idx, tok in enumerate(tokens[:-1]):
            ids.extend(tok)
            ids.append('[SEP]')
            tids.extend([tcache] * (len(tok) + 1))
            sids.extend([scache] * (len(tok) + 1))
            scache = 0 if scache == 1 else 1
            tcache = 0
        tcache = 1
        ids.pop()
        sids.pop()
        tids.pop()
        ids = self.vocab.convert_tokens_to_ids(ids)
        rids = self.vocab.convert_tokens_to_ids(tokens[-1])
        trids = [tcache] * len(rids)
        srids = [scache] * len(rids)
        truncate_pair_with_other_ids(ids, rids, tids, trids, sids, srids, self.args['max_len'])
        ids = [self.cls] + ids + [self.sep] + rids + [self.sep]
        tids = [0] + tids + [0] + trids + [1]
        sids = [0] + sids + [sids[-1]] + srids + [srids[-1]]
        assert len(ids) == len(tids) == len(sids)
        return ids, tids, sids
