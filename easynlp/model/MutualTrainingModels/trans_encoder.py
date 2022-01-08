from model.utils import *
from config import load_model_config
from .dual_bert_mutual import *
from .sa_bert_mutual import *

class TransEncoder(nn.Module):

    def __init__(self, vocab, **args):
        super(TransEncoder, self).__init__()
        self.args = args
        self.vocab = vocab
        self.sep, self.cls, self.pad, self.eos = self.vocab.convert_tokens_to_ids([
            '[SEP]', '[CLS]', '[PAD]', '[EOS]'
        ])

        # load bi-encoder model
        config = deepcopy(self.args)
        config.update(load_model_config(self.args['bi_encoder'], self.args['mode']))
        self.bi_encoder_model = BERTDualMutualEncoder(**config)
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

    def swap_training_model(self):
        if self.training_model == 'bi-encoder':
            self.bi_encoder_model.eval()
            self.cross_encoder_model.train()
            self.training_model = 'cross-encoder'
        else:
            self.cross_encoder_model.eval()
            self.bi_encoder_model.train()
            self.training_model = 'bi-encoder'
    
    @torch.no_grad()
    def obtain_training_batch_for_bi_encoder(self, batch):
        '''cross-encoder for soft labeling'''
        context, response, candidates = deepcopy(batch['context']), deepcopy(batch['response']), deepcopy(batch['candidates'])
        topk = self.args['gray_cand_num']
        candidates = list(chain(*candidates))
        context = list(chain(*[[i]*topk for i in context]))
        ids, cids = self.text_to_tensor_bi_encoder(context, candidates)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        cids_mask = generate_mask(cids)
        ids, cids, ids_mask, cids_mask = to_cuda(ids, cids, ids_mask, cids_mask)

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
        return {
            'ids': ids,
            'ids_mask': ids_mask,
            'rids': cids,
            'rids_mask': cids_mask,
            'soft_label': soft_label,
        }

    @torch.no_grad()
    def obtain_training_batch_for_cross_encoder(self, batch):
        '''bi-encoder for soft labeling'''
        context, response, candidates = deepcopy(batch['context']), deepcopy(batch['response']), deepcopy(batch['candidates'])
        topk = len(candidates[0])
        candidates = list(chain(*candidates))
        context = list(chain(*[[i]*topk for i in context]))

        ids, sids, tids = self.text_to_tensor_cross_encoder(context, candidates)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, sids, tids, ids_mask = to_cuda(ids, sids, tids, ids_mask)
        
        # for bi-encoder labeling
        ids_, cids_ = self.text_to_tensor_bi_encoder(context, candidates)
        ids_ = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
        cids_ = pad_sequence(cids_, batch_first=True, padding_value=self.pad)
        ids_mask_ = generate_mask(ids_)
        cids_mask_ = generate_mask(cids_)
        ids_, cids_, ids_mask_, cids_mask_ = to_cuda(ids_, cids_, ids_mask_, cids_mask_)
        soft_label = self.bi_encoder_predict({
            'ids': ids_,
            'ids_mask': ids_mask_,
            'rids': cids_,
            'rids_mask': cids_mask_,
        })    # [B*K]
        return {
            'ids': ids,
            'sids': sids,
            'tids': tids,
            'mask': ids_mask,
            'soft_label': soft_label
        }

    def forward(self, batch, scaler, optimizer):
        tloss, counter = 0, 0
        if self.training_model == 'bi-encoder':
            batches = self.obtain_training_batch_for_bi_encoder(batch)
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
                sub_label = batches['soft_label'][i:i+self.args['inner_bsz']]
                sub_ids_mask = generate_mask(sub_ids)
                sub_rids_mask = generate_mask(sub_rids)
                sub_ids, sub_rids, sub_ids_mask, sub_rids_mask = to_cuda(sub_ids, sub_rids, sub_ids_mask, sub_rids_mask)
                batch = {
                    'ids': sub_ids,
                    'ids_mask': sub_ids_mask,
                    'rids': sub_rids,
                    'rids_mask': sub_rids_mask,
                    'soft_label': sub_label,
                }
                with autocast():
                    loss = self.bi_encoder_model(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.parameters(), self.args['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                tloss += loss
                counter += 1
        else:
            batches = self.obtain_training_batch_for_cross_encoder(batch)
            for i in range(0, len(batches['ids']), self.args['inner_bsz']):
                sub_ids = pad_sequence(
                    batches['ids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_tids = pad_sequence(
                    batches['tids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_sids = pad_sequence(
                    batches['sids'][i:i+self.args['inner_bsz']],
                    batch_first=True,
                    padding_value=self.pad,
                )
                sub_label = batches['soft_label'][i:i+self.args['inner_bsz']]
                sub_ids_mask = generate_mask(sub_ids)
                sub_ids, sub_tids, sub_sids, sub_ids_mask = to_cuda(sub_ids, sub_tids, sub_sids, sub_ids_mask)
                batch = {
                    'ids': sub_ids,
                    'sids': sub_sids,
                    'tids': sub_tids,
                    'mask': sub_ids_mask,
                    'soft_label': sub_label
                }
                with autocast():
                    loss = self.cross_encoder_model(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.parameters(), self.args['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                tloss += loss
                counter += 1
        tloss /= counter
        return tloss

    @torch.no_grad()
    def predict(self, batch, test_model_name='bi-encoder'):
        context, responses = batch['context'], batch['responses']
        context = [context] * 10
        if test_model_name == 'bi-encoder':
            ids, rids = self.text_to_tensor_bi_encoder(context, responses)
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
        else:
            ids, tids, sids = self.text_to_tensor_cross_encoder(context, responses)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, tids, sids, ids_mask = to_cuda(ids, tids, sids, ids_mask)
            return self.cross_encoder_predict({
                'ids': ids,
                'tids': tids,
                'sids': sids,
                'mask': ids_mask,
            })

    def text_to_tensor_bi_encoder(self, ctx, res):
        ids, ids_mask, rids, rids_mask = [], [], [], []
        for c, r in zip(ctx, res):
            items = self.vocab.batch_encode_plus(c+[r], add_special_tokens=False)['input_ids']
            cids, rid = items[:-1], items[-1]
            ids_ = []
            for u in cids:
                ids_.extend(u + [self.sep])
            ids_.pop()
            ids_ = [self.cls] + ids_[-self.args['max_len']+2:] + [self.sep]
            ids.append(torch.LongTensor(ids_))
            rids.append(
                torch.LongTensor(
                    [self.cls] + rid[:self.args['res_max_len']-2] + [self.sep]
                )
            )
        return ids, rids

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
