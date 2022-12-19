from model.utils import *
from .gpt2_original import GPT2OriginalModel, GPT2wt103Model
from .knn_lm import KNNLMModel
from model.RepresentationModels import DensePhraseEncoder, DensePhraseV2Encoder, DensePhraseV3Encoder, DensePhraseV4Encoder, DensePhraseV7Encoder, FastDensePhraseV8Encoder, FastDensePhraseV10Encoder, FastDensePhraseV13Encoder, FastDensePhraseV15Encoder, FastDensePhraseV16Encoder, FastDensePhraseV17Encoder, FastDensePhraseV22Encoder, FastDensePhraseV11Encoder, FastDensePhraseV25Encoder, FastDensePhraseV26Encoder, FastDensePhraseV27Encoder, Copyisallyouneed, FastDensePhraseV28Encoder, FastDensePhraseV29Encoder
from .utils import *
from config import *

class CopyGenerationEncoder(nn.Module):

    def __init__(self, **args):
        super(CopyGenerationEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        # gpt2 baseline
        generator_args = deepcopy(self.args)
        generator_args['model'] = 'gpt2-original'
        config = load_config(generator_args)
        generator_args.update(config)
        self.generator = GPT2OriginalModel(**generator_args) 
        # self.generator = GPT2wt103Model(**generator_args) 

        # gpt2 en-wiki fine-tuned baseline
        # '''
        generator_args = deepcopy(self.args)
        generator_args['model'] = 'gpt2-original'
        generator_args['dataset'] = 'en_wiki'
        config = load_config(generator_args)
        generator_args.update(config)
        self.en_wiki_generator = GPT2OriginalModel(**generator_args) 
        # '''

        # knn-lm gpt2 baseline
        '''
        generator_args = deepcopy(self.args)
        generator_args['model'] = 'knn-lm'
        generator_args['dataset'] = 'wikitext103'
        config = load_config(generator_args)
        generator_args.update(config)
        # self.wikitext103_knn_lm_generator = KNNLMModel(**generator_args) 
        self.knn_lm_generator = KNNLMModel(**generator_args) 
        print(f'[!] load the knn-lm model over ...')
        ## init the knn-lm searcher
        faiss_searcher_args = deepcopy(self.args)
        faiss_searcher_args['model'] = 'knn-lm'
        faiss_searcher_args['dataset'] = 'wikitext103'
        config = load_config(faiss_searcher_args)
        faiss_searcher_args.update(config)
        faiss_searcher = Searcher(
            faiss_searcher_args['index_type'],
            dimension=faiss_searcher_args['dimension'],
            nprobe=faiss_searcher_args['index_nprobe']
        )
        pretrained_model_name = faiss_searcher_args['pretrained_model'].replace('/', '_')
        model_name = faiss_searcher_args['model']
        faiss_searcher.load(
            f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',        
            f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',        
        )
        # self.wikitext103_knn_lm_generator.init_searcher(faiss_searcher)
        self.knn_lm_generator.init_searcher(faiss_searcher)
        print(f'[!] wikitext103 knn-lm generator baseline init over ...')
        '''
        
        '''
        generator_args = deepcopy(self.args)
        generator_args['model'] = 'knn-lm'
        config = load_config(generator_args)
        generator_args.update(config)
        self.knn_lm_generator = KNNLMModel(**generator_args) 
        print(f'[!] load the knn-lm model over ...')
        ## init the knn-lm searcher
        faiss_searcher_args = deepcopy(self.args)
        faiss_searcher_args['model'] = 'knn-lm'
        config = load_config(faiss_searcher_args)
        faiss_searcher_args.update(config)
        faiss_searcher = Searcher(
            faiss_searcher_args['index_type'],
            dimension=faiss_searcher_args['dimension'],
            nprobe=faiss_searcher_args['index_nprobe']
        )
        pretrained_model_name = faiss_searcher_args['pretrained_model'].replace('/', '_')
        model_name = faiss_searcher_args['model']
        faiss_searcher.load(
            f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',        
            f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',        
        )
        self.knn_lm_generator.init_searcher(faiss_searcher)
        print(f'[!] en-wiki knn-lm generator baseline init over ...')
        '''

        # our proposed phrase-gpt2 model
        retriever_args = deepcopy(self.args)
        retriever_args['model'] = 'phrase-copy'
        config = load_config(retriever_args)
        retriever_args.update(config)
        # self.retriever = DensePhraseV7Encoder(**retriever_args) 
        # self.retriever = DensePhraseV4Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV11Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV13Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV15Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV16Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV17Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV22Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV25Encoder(**retriever_args) 
        
        # self.retriever = FastDensePhraseV29Encoder(**retriever_args) 
        # self.retriever = FastDensePhraseV27Encoder(**retriever_args) 
        self.retriever = Copyisallyouneed(**retriever_args)
        self.test_max_len = self.args['test_max_len']

        if self.args['lang'] == 'en':
            # self.process_documents = self.process_documents_en_v2
            # self.process_documents = self.process_documents_en_v3
            # self.process_documents = self.process_documents_en_v5
            # self.process_documents = self.process_documents_en_v7
            # self.process_documents = self.process_documents_en_v5_gpt2
            # self.process_documents = self.process_documents_en_v6
            # self.process_documents = self.process_documents_en_v4
            
            # self.process_documents = self.process_documents_en_v8
            self.process_documents = self.process_documents_en_v9
            self.nlp = spacy.load('en_core_web_sm')
            # self.punc_set = set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '-', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', 'of', 'in', 'to', 'during', 'at', 'and', 'or', 'but', 'that', 'if', 'while', 'which', 'whose', 'for', 'as', 'from', 'to', 'a', 'an', 'the', 'The', 'That', 'on', 'by', 'is', 'has', 'will', 'are', 'have', 'was', 'had', 'should', 'must', 'about', 'with', 'after', 'when', 'before', 'into', 'within', 'than'])
            # self.punc_set |= set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '-', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}'])
            # self.punc_set = set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}', ' that', ' which', ' when', ' after', ' before', ' while', ' during', ' with', ' than', ' within'])
            self.punc_set = set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}', 'as', ' as'])
            self.punc_set |= set([' ,', ' .', ' "', " '", ' ?', ' !', ' @', ' -', ' <', ' >', ' :', ' ;', ' /', ' _', ' +', ' =', ' ~', ' `', ' #', ' $', ' %', ' ^', ' &', ' *', ' (', ' )', ' [', ' ]', ' {', ' }', '/'])
            # in gpt2_english
            self.blank_token_gpt2 = self.retriever.tokenizer.convert_ids_to_tokens(220)
        else:
            # self.process_documents = self.process_documents_v2
            self.process_documents = self.process_documents_v3
            # self.process_documents = self.process_documents_v6
            # self.process_documents = self.process_documents_v5
            # self.process_documents = self.process_documents_v4
            self.punc_set = set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '-', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}', '，', '。', '！', '“', '”', '？', '@', '《', '》', '：', '；', '`', '（', '）'])

        # self.retrieval_generation_search = self.retrieval_generation_search_fast_dynamic_search
        # self.process_documents = self.process_documents_en_v5_fast
        self.retrieve_doc = self.retrieve_doc_bm25

    def init_searcher_agent(self, agent):
        self.search_agent = agent
        print(f'[!] init the searcher agent over')

    def init_searcher(self, searcher, base_data):
        self.searcher = searcher  
        self.base_data = base_data
        print(f'[!] init the simcse search agent over')

    def init_searcher_en_wiki(self, searcher, base_data):
        self.en_wiki_searcher = searcher  
        self.en_wiki_base_data = base_data
        print(f'[!] init the EN-WIKI simcse search agent over')

    def init_faiss_searcher(self, searcher):
        self.faiss_searcher = searcher
        # build the token embeddings into the faiss index 
        embd = F.normalize(self.retriever.token_embeddings, dim=-1).detach().cpu().numpy()
        text = self.retriever.tokenizer.convert_ids_to_tokens(range(len(self.retriever.tokenizer)))
        self.faiss_searcher.add(embd, text)
        print(f'[!] init the faiss searcher agent over, faiss index size: {self.faiss_searcher.searcher.ntotal}')

    @torch.no_grad()
    def search_from_words(self, query, search_topk=5):
        self.retriever.eval()
        # dp = torch.matmul(query, F.normalize(self.retriever.token_embeddings, dim=-1).t()).squeeze(0)
        dp = torch.matmul(query, self.retriever.token_embeddings.t()).squeeze(0)
        dis, topk = dp.topk(search_topk, dim=-1)
        dis = dis.tolist()
        topk = topk.tolist()
        if self.args['lang'] == 'zh':
            candidates = [(self.retriever.tokenizer.convert_ids_to_tokens(i), round(d, 4)) for i, d in zip(topk, dis)]
        else:
            candidates = [(self.retriever.tokenizer.decode(i), round(d, 4), '') for i, d in zip(topk, dis)]
        return candidates

    @torch.no_grad()
    def search_from_documents(self, query, phrase_reps, phrase_source, search_topk=5, head_weight=1.0, tail_weight=1.0):
        self.retriever.eval()
        dp = torch.matmul(query, phrase_reps.t()).squeeze(0)   

        search_num = min(search_topk, len(phrase_reps))
        dis, topk = dp.topk(search_num, dim=-1)    # [K]
        dis = dis.tolist()
        if self.args['lang'] == 'zh':
            candidates = [(''.join(phrase_source[i][-1].split()), round(d, 4)) for i, d in zip(topk, dis)]
        else:
            # candidates = [(phrase_source[i][-1], round(d, 4) * head_weight) if phrase_source[i][0] == -1 else (phrase_source[i][-1], round(d, 4)) for i, d in zip(topk, dis)]
            candidates = [(phrase_source[i][-2], round(d, 4), phrase_source[i][-1]) for i, d in zip(topk, dis)]
        return candidates

    def search_from_faiss(self, query, search_topk=5):
        candidates, distance = self.faiss_searcher._search_dis(query.cpu().numpy(), topk=search_topk)
        candidates, distance = candidates[0], [i.item() for i in distance[0]]
        new_candidates = []
        new_candidates_pool = set()
        for c, d in zip(candidates, distance):
            if c in new_candidates_pool:
                continue
            new_candidates_pool.add(c)
            new_candidates.append((c, d))
        return new_candidates

    @torch.no_grad()
    def process_documents_en(self, documents):
        self.retriever.eval()

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            return True

        # init
        characters = set("\".,!?`[]{}'';:><+=-_&^%$#@()/–\\")
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments, segments_label = [], []
            for seg in self.nlp(doc):
                # if _check_valid(seg.text) and seg.pos_ not in ['CCONJ', 'DET', 'SCONJ', 'PRON', 'ADP', 'PUNCT', 'SYM', 'X', 'AUX']:
                if _check_valid(seg.text) and seg.pos_ not in ['CCONJ', 'ADP']:
                    segments_label.append(1)
                else:
                    segments_label.append(0)
                segments.append(seg.text)
            if segments:
                seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']
            else:
                continue

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
            phrase_sources.extend([(s, e, ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        # phrase_reps = torch.cat(phrase_reps)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        print(f'[!] collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents(self, documents):
        '''TODO: add the sentence as the phrase set'''
        self.retriever.eval()

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            for w in black_words:
                if w in string:
                    return False
            return True

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?`[]{}'';:><+=-_&^%$#@《》/\\")
        black_words = ['编辑', '人物', '生平', '背景', '死因', '之谜', '简介', '图片', '来源', '记录', '经历', '演艺经历', '参考资料', '版本', '演员表', '简体名', '作品时间', '剧名类型', '个人成就', '角色介绍', '个人资料', '英文名', '参考', '履历', '图示' ,'业务范围', '时刻表']
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments = list(jieba.cut(doc))
            segments_label = []
            for seg in segments:
                if _check_valid(seg):
                    segments_label.append(1)
                else:
                    segments_label.append(0)
            if segments:
                seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']
            else:
                continue

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    b_ = len(cache_dids)
                    p_index = index
                    p_cache_dids = deepcopy(cache_dids)
                    while p_index < len(doc) and label[p_index] == 1:
                        p_cache_dids.extend(doc[p_index])
                        if min_length <= len(p_cache_dids) - b_ <= max_length:
                            phrases.append((b_, len(p_cache_dids) - 1))
                        p_index += 1
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            rep = torch.cat([self.retriever.s_proj(s_rep), self.retriever.e_proj(e_rep)], dim=-1)
            phrase_reps.append(rep)
            phrase_sources.extend([(s, e, self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        phrase_reps = torch.cat(phrase_reps)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources

    def truncation(self, a, b, max_len):
        aa, bb = deepcopy(a), deepcopy(b)
        while True:
            if len(aa) + len(bb) <= max_len:
                break
            else:
                aa.pop(0)
        return aa, bb

    @torch.no_grad()
    def retrieval_generation_search_e2e(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        ids = batch['ids']
        batch_size, seqlen = ids.size()
        generated = []
        while len(ids[0]) < seqlen + self.test_max_len:
            # init the query
            query = self.retriever.get_query_rep(ids)
            # search candidate phrases
            candidates = self.search_from_faiss(query, search_topk=beam_width)
            if self.args['lang'] == 'zh':
                candidates = [c for c in candidates if '[UNK]' not in c[0]]
            else:
                candidates = [c for c in candidates if '<|endoftext|>' not in c[0]]
            candidates = sorted(candidates, key=lambda x:x[1], reverse=False)
            candidate_tokens = [item[0] for item in candidates]
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()

            ids, candidate = self.decoding_one_step_inner(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            if self.args['lang'] == 'zh':
                generated.append(f'{candidate} ')
            else:
                # generated.append(f'[{candidate}] ')
                generated.append(ids[0, -1].item())
        if self.args['lang'] == 'zh':
            generated = ''.join(generated)
        else:
            generated = self.retriever.bert_tokenizer.decode(generated)
        return generated

    @torch.no_grad()
    def retrieval_generation_search_e2e_onestep(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        ids = batch['ids']
        _, prefix_length = ids.size()
        # init the phrases
        batch_size, seqlen = ids.size()
        query = self.retriever.get_query_rep(ids)
        candidates = self.search_from_faiss(query, search_topk=beam_width)
        candidates = [c for c in candidates if '[UNK]' not in c[0]]
        candidates = sorted(candidates, key=lambda x:x[1], reverse=False)
        return candidates

    @torch.no_grad()
    def retrieval_generation_search_one_step(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        head_weight, tail_weight = batch['head_weight'], batch['tail_weight']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        _, prefix_length = ids.size()
        # init the phrases
        # phrase_reps, phrase_sources = self.process_documents(doc)
        batch_size, seqlen = ids.size()
        query = self.retriever.get_query_rep(ids)
        candidates = self.search_from_documents(query, phrase_reps, phrase_sources, search_topk=beam_width, head_weight=head_weight, tail_weight=tail_weight)
        # candidates = self.search_from_words(query, search_topk=beam_width)
        if self.args['lang'] == 'zh':
            candidates = [c for c in candidates if '[UNK]' not in c[0]]
        else:
            # candidates = [[' ' + i[0], i[1]] for i in candidates]
            candidates = [c for c in candidates if '<|endoftext|>' not in c[0] and 'unk' not in c[0]]
        alpha, beta = self.args['coarse_score_alpha'], 1 - self.args['coarse_score_alpha']
        candidate_tokens = [c[0] for c in candidates]
        candidate_prob = self.retriever.fast_rerank(ids, candidate_tokens).tolist()
        candidates = [[t[0], t[1] * alpha + beta * s] for t, s in zip(candidates, candidate_prob)]
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
        return candidates

    @torch.no_grad()
    def retrieval_generation_search_v2(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        head_weight, tail_weight = batch['head_weight'], batch['tail_weight']
        alpha = batch['coarse_score_alpha']
        beta = 1 - alpha
        coarse_score_softmax_temp = batch['coarse_score_softmax_temp']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        _, prefix_length = ids.size()

        # init the phrases
        if batch['use_phrase_cache'] is False:
            self.cache_tokens, self.cache_start_point_matrix, self.cache_end_point_matrix, self.cache_opt_start_point_matrix, self.cache_opt_end_point_matrix = process_documents(doc)
        else:
            print(f'[!] load {len(phrase_reps)} cached phrase to save the process time')

        batch_size, seqlen = ids.size()
        generated = []
        inputs_embeds = []
        while len(ids[0]) < seqlen + self.test_max_len:
            query = self.retriever.get_query_rep(input_embeds)
            candidates, ipt_phrase_reps = self.search_from_documents(query, phrase_reps, phrase_sources, search_topk=beam_width)
            # word_candidates = self.search_from_words(query, search_topk=beam_width)

            if self.args['lang'] == 'zh':
                candidates = [c for c in candidates if '[UNK]' not in c[0]]
            else:
                new_candidates = [c for c in candidates if '<|endoftext|>' not in c[0] and 'unk' not in c[0]]
                if len(new_candidates) > 0:
                    candidates = new_candidates
                    
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
            candidate_prob = F.softmax(candidate_prob/self.args['softmax_temp'], dim=-1)
            candidate_tokens = [item[0] for item in candidates]
            
            debug_info = 0
            if debug_info == 1:
                # rerank pipeline
                # alpha, beta = self.args['coarse_score_alpha'], 1 - self.args['coarse_score_alpha']
                candidate_tokens = [item[0] for item in candidates]
                candidate_prob = self.retriever.fast_rerank(ids, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                # candidate_prob = self.retriever.fast_rerank_v2(query, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                candidates = [[item[0], item[1] * alpha + s * beta] for item, s in zip(candidates, candidate_prob)]
                candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
                candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
                candidate_prob = F.softmax(candidate_prob / self.args['softmax_temp'], dim=-1)
                candidate_tokens = [item[0] for item in candidates]

            ids, candidate = self.decoding_one_step_inner(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            if self.args['lang'] == 'zh':
                generated.append(f'{candidate} ')
            else:
                # generated.append(f'[{candidate}] ')
                generated.append(candidate)
        generated = ''.join(generated)
        return generated
    
    @torch.no_grad()
    def retrieval_generation_search(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        head_weight, tail_weight = batch['head_weight'], batch['tail_weight']
        alpha = batch['coarse_score_alpha']
        beta = 1 - alpha
        coarse_score_softmax_temp = batch['coarse_score_softmax_temp']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        _, prefix_length = ids.size()

        # init the phrases
        if batch['use_phrase_cache'] is False:
            if self.args['lang'] == 'en':
                phrase_reps, phrase_sources = self.process_documents_en_v9(doc)
                # phrase_reps, phrase_sources = self.process_documents_en_v10(doc)
                # phrase_reps, phrase_sources = self.process_documents_en_v5(doc)
            else:
                phrase_reps, phrase_sources = self.process_documents_v6([doc[0]])
                phrase_reps_, phrase_sources_ = self.process_documents(doc[1:])
                phrase_reps = torch.cat([phrase_reps, phrase_reps_], dim=0)
                phrase_sources.extend(phrase_sources_)
            self.cache_phrase_reps, self.cache_phrase_sources = phrase_reps, phrase_sources
        else:
            phrase_reps, phrase_sources = self.cache_phrase_reps, self.cache_phrase_sources
            print(f'[!] load {len(phrase_reps)} cached phrase to save the process time')

        batch_size, seqlen = ids.size()
        generated = []
        is_phrase_label = []
        phrase_length_avg = []
        bt = time.time()
        while len(ids[0]) < seqlen + self.test_max_len:
            query = self.retriever.get_query_rep(ids)
            # if self.retriever.tokenizer.decode(ids[0, -1]) in self.punc_set:
            if 0:
                candidates = self.search_from_words(query, search_topk=beam_width)
            else:
                candidates = self.search_from_documents(query, phrase_reps, phrase_sources, search_topk=beam_width, head_weight=head_weight, tail_weight=tail_weight)
            # word_candidates = self.search_from_words(query, search_topk=beam_width)
            # combination of the two set of the candidates
            # candidates += [(string, score * head_weight, context) for string, score, context in word_candidates]

            if self.args['lang'] == 'zh':
                candidates = [c for c in candidates if '[UNK]' not in c[0]]
            else:
                new_candidates = [c for c in candidates if '<|endoftext|>' not in c[0] and 'unk' not in c[0]]
                if len(new_candidates) > 0:
                    candidates = new_candidates
                    
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
            candidate_prob = F.softmax(candidate_prob / self.args['softmax_temp'], dim=-1)
            candidate_prob = F.softmax(candidate_prob, dim=-1)
            candidate_tokens = [item[0] for item in candidates]
            candidate_is_phrase_label = [0 if item[-1] == '' else 1 for item in candidates]
            
            debug_info = 0
            if debug_info == 1:
                # rerank pipeline
                # alpha, beta = self.args['coarse_score_alpha'], 1 - self.args['coarse_score_alpha']
                candidate_tokens = [item[0] for item in candidates]
                candidate_prob = self.retriever.fast_rerank(ids, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                # candidate_prob = self.retriever.fast_rerank_v2(query, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                candidates = [[item[0], item[1] * alpha + s * beta, item[2]] for item, s in zip(candidates, candidate_prob)]
                candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
                candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
                # candidate_prob = F.softmax(candidate_prob / self.args['softmax_temp'], dim=-1)
                candidate_prob = F.softmax(candidate_prob, dim=-1)
                candidate_tokens = [item[0] for item in candidates]

            before_length = len(ids[0])
            ids, candidate, index = self.decoding_one_step_inner(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            phrase_length = len(ids[0]) - before_length
            phrase_length_avg.append(phrase_length)
            print(candidates[index])
            if self.args['lang'] == 'zh':
                generated.append(f'{candidate} ')
            else:
                # generated.append(f'[{candidate}] ')
                generated.append(candidate)
            is_phrase_label.append(candidate_is_phrase_label[index])
            
            # if (len(ids[0]) - seqlen) % update_step == 0:
            #     if self.args['lang'] == 'zh':
            #         string = ''.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     else:
            #         string = ' '.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     ipdb.set_trace()
            #     doc = self.retrieve_doc(string, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            #     phrase_reps, phrase_sources = self.process_documents(doc)

            # re-size the length of the ids
            # ids = ids[:, -500:]
        time_cost = time.time() - bt
        generated = ''.join(generated)
        # return generated, np.mean(is_phrase_label), time_cost
        # return generated, np.mean(phrase_length_avg), time_cost
        return generated, len(phrase_reps), time_cost

    def decoding_one_step_inner(self, ids, candidates, candidates_prob, generation_method, topk=1., topp=1., model_prediction_confidence=0.4):
        if generation_method == 'contrastive-search':
            candidates_ = self.retriever.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            ids, index = ContrastiveDecodingOneStepUnify(
                self.retriever, 
                ids, 
                candidates_,
                model_prediction_confidence, 
                self.retriever.pad,
                self.args['temperature']
            )
            candidate = candidates[index]
        elif generation_method == 'greedy-search':
            index = 0
            candidate = candidates[index]
            sub_ids = self.retriever.tokenizer.encode(candidate, add_special_tokens=False)
            sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
            ids = torch.cat((ids, sub_ids), dim=-1)
        elif generation_method == 'nucleus-search':
            new_scores = top_k_top_p_filtering(candidates_prob, top_k=topk, top_p=topp)
            index = torch.multinomial(F.softmax(new_scores, dim=-1), num_samples=1).squeeze(-1)
            candidate = candidates[index.item()]
            sub_ids = self.retriever.tokenizer.encode(candidate, add_special_tokens=False)
            sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
            ids = torch.cat((ids, sub_ids), dim=-1)
        else:
            raise Exception(f'[!] Unknow generation method: {generation_method}')
        return ids, candidate, index

    @torch.no_grad()
    def retrieval_generation_search_fast_dynamic_search(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        head_weight, tail_weight = batch['head_weight'], batch['tail_weight']
        alpha = batch['coarse_score_alpha']
        beta = 1 - alpha
        coarse_score_softmax_temp = batch['coarse_score_softmax_temp']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        _, prefix_length = ids.size()

        # init the phrases
        if batch['use_phrase_cache'] is False:
            begin_rep, end_rep, phrase_sources, collected_docids = self.process_documents(doc)
            self.cache_begin_rep, self.cache_end_rep, self.cache_phrase_sources, self.cache_collected_docids = begin_rep, end_rep, phrase_sources, collected_docids
        else:
            begin_rep, end_rep, phrase_sources, collected_docids = self.cache_begin_rep, self.cache_end_rep, self.cache_phrase_sources, self.cache_collected_docids
            print(f'[!] load {len(phrase_reps)} cached phrase to save the process time')

        batch_size, seqlen = ids.size()
        generated = []
        while len(ids[0]) < seqlen + self.test_max_len:
            query = self.retriever.get_query_rep(ids)

            if self.retriever.tokenizer.decode(ids[0, -1]) in self.punc_set:
                candidates = self.search_from_words(query)
            else:
                candidates = self.search_from_documents_fast_dynamic(query, begin_rep, end_rep, phrase_sources, collected_docids, search_topk=beam_width)

            if self.args['lang'] == 'zh':
                candidates = [c for c in candidates if '[UNK]' not in c[0]]
            else:
                new_candidates = [c for c in candidates if '<|endoftext|>' not in c[0] and 'unk' not in c[0]]
                if len(new_candidates) > 0:
                    candidates = new_candidates
                    
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
            candidate_prob = F.softmax(candidate_prob / self.args['softmax_temp'], dim=-1)
            candidate_tokens = [item[0] for item in candidates]
            
            debug_info = 0
            if debug_info == 1:
                # rerank pipeline
                # alpha, beta = self.args['coarse_score_alpha'], 1 - self.args['coarse_score_alpha']
                candidate_tokens = [item[0] for item in candidates]
                candidate_prob = self.retriever.fast_rerank(ids, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                # candidate_prob = self.retriever.fast_rerank_v2(query, candidate_tokens, temp=coarse_score_softmax_temp).tolist()
                candidates = [[item[0], item[1] * alpha + s * beta] for item, s in zip(candidates, candidate_prob)]
                candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
                candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
                candidate_prob = F.softmax(candidate_prob / self.args['softmax_temp'], dim=-1)
                candidate_tokens = [item[0] for item in candidates]

            ids, candidate = self.decoding_one_step_inner(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            if self.args['lang'] == 'zh':
                generated.append(f'{candidate} ')
            else:
                generated.append(f'[{candidate}] ')
                # generated.append(candidate)
            
            # if (len(ids[0]) - seqlen) % update_step == 0:
            #     if self.args['lang'] == 'zh':
            #         string = ''.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     else:
            #         string = ' '.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     doc = self.retrieve_doc(string, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            #     phrase_reps, phrase_sources = self.process_documents(doc)
        generated = ''.join(generated)
        return generated

    @torch.no_grad()
    def work(self, data):

        '''generation_method: nucleus-search, greedy-search, contrastive-search'''

        document = data['document']
        decoding_method = data['decoding_method'] 
        generation_method = data['generation_method']
        topk, topp, beam_width, model_prediction_confidence = data['topk'], data['topp'], data['beam_width'], data['model_prediction_confidence']
        phrase_alpha = data['phrase_alpha']
        update_step = data['update_step']
        head_weight, tail_weight = data['head_weight'], data['tail_weight']
        use_phrase_cache = data['use_phrase_cache']
        beam_search_size = data['beam_search_size']
        assert generation_method in ['contrastive-search', 'greedy-search', 'nucleus-search']
        prefix = data['prefix']
        ground_truth = data['ground_truth']
        max_gen_len = data['max_gen_len']
        self.args['softmax_temp'] = data['softmax_temp']
        print(f'[!] set softmax temperature as {self.args["softmax_temp"]}')

        

        if decoding_method == 'knn-lm-topk-topp-search':

            ids = self.knn_lm_generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.knn_lm_generator.test_max_len = max_gen_len
            self.knn_lm_generator.topk, self.knn_lm_generator.topp = topk, topp
            self.knn_lm_generator.args['temp'] = data['temp']
            print(f'[!] set the knn-lm temperature: {self.knn_lm_generator.args["temp"]}')
            bt = time.time()
            response = self.knn_lm_generator.topk_topp_search(batch)
            time_cost = time.time() - bt
            phrase_ratio = 0.
            pass
        elif decoding_method == 'wikitext103-knn-lm-topk-topp-search':
            ids = self.wikitext103_knn_lm_generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.wikitext103_knn_lm_generator.test_max_len = max_gen_len
            self.wikitext103_knn_lm_generator.topk, self.wikitext103_knn_lm_generator.topp = topk, topp
            self.wikitext103_knn_lm_generator.args['temp'] = data['temp']
            print(f'[!] set the knn-lm temperature: {self.wikitext103_knn_lm_generator.args["temp"]}')
            response = self.wikitext103_knn_lm_generator.topk_topp_search(batch)
            phrase_ratio = 0.
        elif decoding_method == 'knn-lm-greedy-search':

            ids = self.knn_lm_generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.knn_lm_generator.test_max_len = max_gen_len
            self.knn_lm_generator.args['temp'] = data['temp']
            print(f'[!] set the knn-lm temperature: {self.knn_lm_generator.args["temp"]}')
            bt = time.time()
            response = self.knn_lm_generator.greedy_search(batch)
            time_cost = time.time() - bt
            phrase_ratio = 0.
            pass
        elif decoding_method == 'en-wiki-topk-topp-search':

            ids = self.en_wiki_generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.en_wiki_generator.test_max_len = max_gen_len
            self.en_wiki_generator.topk, self.en_wiki_generator.topp = topk, topp
            response = self.en_wiki_generator.topk_topp_search(batch)
            phrase_ratio = 0.
            pass
        elif decoding_method == 'en-wiki-greedy-search':

            ids = self.en_wiki_generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }
            self.en_wiki_generator.test_max_len = max_gen_len
            response = self.en_wiki_generator.greedy_search(batch)
            pass
        
        elif decoding_method == 'topk-topp-search':

            ids = self.generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.generator.test_max_len = max_gen_len
            self.generator.topk, self.generator.topp = topk, topp
            bt = time.time()
            response = self.generator.topk_topp_search(batch)
            time_cost = time.time() - bt
            phrase_ratio = 0.
            pass
        elif decoding_method == 'greedy-search':

            ids = self.generator.vocab.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.generator.test_max_len = max_gen_len
            bt = time.time()
            response = self.generator.greedy_search(batch)
            time_cost = time.time() - bt
            phrase_ratio = 0.
            pass
        elif decoding_method == 'contrastive-search':
            self.generator.test_max_len = max_gen_len
            response = self.generator.contrastive_search(batch)
            pass
        elif decoding_method == 'beam-search':
            self.generator.test_max_len = max_gen_len
            response = self.generator.beam_search(batch)
            pass
        elif decoding_method == 'word-nucleus-search':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }
            # batch['test_max_len'] = self.test_max_len
            batch['test_max_len'] = max_gen_len
            batch['topk'] = topk
            batch['topp'] = topp
            response = self.retriever.nucleus_search(batch)
        elif decoding_method == 'word-greedy-search':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }
            # batch['test_max_len'] = self.test_max_len
            batch['test_max_len'] = max_gen_len
            response = self.retriever.greedy_search(batch)
            phrase_ratio = 0.
        elif decoding_method == 'word-contrastive-search':
            # batch['test_max_len'] = self.test_max_len
            batch['test_max_len'] = max_gen_len
            batch['beam_width'] = beam_width
            batch['model_prediction_confidence'] = model_prediction_confidence
            response = self.retriever.contrastive_search(batch)
        elif decoding_method == 'retrieval-generation-search-onestep':
            docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'docs': docs, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
            }
            response = self.retrieval_generation_search_one_step(batch)
        elif decoding_method == 'retrieval-generation-search-en-wiki':

            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.current_searcher = self.en_wiki_searcher
            self.current_base_data = self.en_wiki_base_data

            self.test_max_len = max_gen_len
            docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            docs = [[prefix]] + docs
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'docs': docs, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
                'use_phrase_cache': use_phrase_cache,
                'head_weight': head_weight,
                'tail_weight': tail_weight,
                'coarse_score_alpha': data['coarse_score_alpha'],
                'coarse_score_softmax_temp': data['coarse_score_softmax_temp']
            }
            response, phrase_ratio, time_cost = self.retrieval_generation_search(batch)
            # response = self.retrieval_generation_search_fast(batch)
        elif decoding_method == 'retrieval-generation-beam-search':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.current_searcher = self.searcher
            self.current_base_data = self.base_data
            self.test_max_len = max_gen_len

            if self.args['recall_topk'] > 0:
                docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            else:
                docs = []
            docs = [[prefix]] + docs

            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'docs': docs, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'beam_search_size': beam_search_size,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
                'use_phrase_cache': use_phrase_cache,
                'head_weight': head_weight,
                'tail_weight': tail_weight,
                'coarse_score_alpha': data['coarse_score_alpha'],
                'coarse_score_softmax_temp': data['coarse_score_softmax_temp']
            }
            response, phrase_ratio = self.retrieval_generation_beam_search(batch)
        elif decoding_method == 'retrieval-generation-search':

            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            # make sure the max generation length is valid during decoding
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids,        
            }

            self.current_searcher = self.searcher
            self.current_base_data = self.base_data
            self.test_max_len = max_gen_len

            if self.args['recall_topk'] > 0:
                docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            else:
                docs = []
            docs = [[prefix]] + docs

            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'docs': docs, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
                'use_phrase_cache': use_phrase_cache,
                'head_weight': head_weight,
                'tail_weight': tail_weight,
                'coarse_score_alpha': data['coarse_score_alpha'],
                'coarse_score_softmax_temp': data['coarse_score_softmax_temp']
            }
            response, phrase_ratio, time_cost = self.retrieval_generation_search(batch)
            # response = self.retrieval_generation_search_fast(batch)
        elif decoding_method == 'retrieval-generation-search-onestep-e2e':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
            }
            response = self.retrieval_generation_search_e2e_onestep(batch)
        elif decoding_method == 'retrieval-generation-search-e2e':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = ids[-512+max_gen_len+2:]
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'phrase_alpha': phrase_alpha, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
            }
            response = self.retrieval_generation_search_e2e(batch)
        else:
            raise Exception(f'[!] Unknow search method: {decoding_method}')

        try:
            return response, phrase_ratio, time_cost
        except:
            return response, phrase_ratio, -1

    def retrieve_doc(self, string, recall_topk=50, max_query_len=512):
        rep = self.search_agent.inference_context_one_sample(string, max_len=max_query_len).cpu().numpy()
        if self.args['dataset'] == 'en_wiki' and self.args['partial'] == 0:
            doc_list = []
        else:
            doc_list = self.current_searcher._search(rep, topk=recall_topk)[0]

        docs = []
        for doc in doc_list:
            if doc not in self.current_base_data:
                continue
            if self.args['lang'] == 'zh':
                string_ = ''.join([item for item in self.current_base_data[doc]])
            else:
                string_ = ' '.join([item for item in self.current_base_data[doc]])
            if string not in string_:
                docs.append(self.current_base_data[doc])
        print(f'[!] collect {len(docs)} documents')
        return docs

    def retrieve_doc_bm25(self, string, recall_topk=50, max_query_len=512):
        if self.args['dataset'] == 'en_wiki' and self.args['partial'] == 0:
            doc_list = []
        else:
            doc_list = self.current_searcher.search(string, topk=recall_topk)

        docs = []
        for doc in doc_list:
            if doc not in self.current_base_data:
                continue
            if self.args['lang'] == 'zh':
                string_ = ''.join([item for item in self.current_base_data[doc]])
            else:
                string_ = ' '.join([item for item in self.current_base_data[doc]])
            if string not in string_:
                docs.append(self.current_base_data[doc])
        print(f'[!] collect {len(docs)} documents')
        return docs

    @torch.no_grad()
    def retrieval_generation_search_fast(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        _, prefix_length = ids.size()
        # init the phrases
        phrase_reps, phrase_sources = self.process_documents(doc)
        batch_size, seqlen = ids.size()
        generated = []
        past_key_values = None
        generation_length = 0
        while generation_length < self.test_max_len:
            past_key_values, query = self.retriever.get_query_rep_fast(ids, past_key_values=past_key_values)
            candidates = self.search_from_documents(query, phrase_reps, phrase_sources, search_topk=beam_width)
            # candidates = self.search_from_words(query, search_topk=beam_width)

            if self.args['lang'] == 'zh':
                candidates = [c for c in candidates if '[UNK]' not in c[0]]
            else:
                candidates = [c for c in candidates if '<|endoftext|>' not in c[0] and 'unk' not in c[0]]

            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()
            candidate_tokens = [item[0] for item in candidates]

            ids, candidate = self.decoding_one_step_inner_fast(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            generation_length += ids.size(-1)
            if self.args['lang'] == 'zh':
                generated.append(f'{candidate} ')
            else:
                # generated.append(f'[{candidate}] ')
                generated.append(candidate)
            
            # if (len(ids[0]) - seqlen) % update_step == 0:
            #     if self.args['lang'] == 'zh':
            #         string = ''.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     else:
            #         string = ' '.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
            #     doc = self.retrieve_doc(string, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            #     phrase_reps, phrase_sources = self.process_documents(doc)
        if self.args['lang'] == 'zh':
            generated = ''.join(generated)
        else:
            generated = ''.join(generated)
        return generated

    def decoding_one_step_inner_fast(self, ids, candidates, candidates_prob, generation_method, topk=1., topp=1., model_prediction_confidence=0.4):
        if generation_method == 'greedy-search':
            candidate = candidates[0]
            sub_ids = self.retriever.tokenizer.encode(candidate, add_special_tokens=False)
            ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        elif generation_method == 'nucleus-search':
            new_scores = top_k_top_p_filtering(candidates_prob, top_k=topk, top_p=topp)
            index = torch.multinomial(F.softmax(new_scores, dim=-1), num_samples=1).squeeze(-1)
            candidate = candidates[index.item()]
            sub_ids = self.retriever.tokenizer.encode(candidate, add_special_tokens=False)
            ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        else:
            raise Exception(f'[!] Unknow generation method: {generation_method}')
        return ids, candidate

    @torch.no_grad()
    def process_documents_v2(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments = doc
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                seg_labels.append(cache_label)
                segment_ids.append(cache)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    b_ = len(cache_dids)
                    p_index = index
                    p_cache_dids = deepcopy(cache_dids)
                    while p_index < len(doc) and label[p_index] == 1:
                        p_cache_dids.extend(doc[p_index])
                        if min_length <= len(p_cache_dids) - b_ <= max_length:
                            phrases.append((b_, len(p_cache_dids) - 1))
                        p_index += 1
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            rep = torch.cat([self.retriever.s_proj(s_rep), self.retriever.e_proj(e_rep)], dim=-1)
            phrase_reps.append(rep)
            phrase_sources.extend([(s, e, self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        phrase_reps = torch.cat(phrase_reps)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v2(self, documents):
        self.retriever.eval()

        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments = doc
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    b_ = len(cache_dids)
                    p_index = index
                    p_cache_dids = deepcopy(cache_dids)
                    while p_index < len(doc) and label[p_index] == 1:
                        p_cache_dids.extend(doc[p_index])
                        if min_length <= len(p_cache_dids) - b_ <= max_length:
                            phrases.append((b_, len(p_cache_dids) - 1))
                        p_index += 1
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
            phrase_sources.extend([(s, e, ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        # phrase_reps = torch.cat(phrase_reps)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources


    @torch.no_grad()
    def process_documents_v3(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            for w in black_words:
                if w in string:
                    return False
            return True

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!?`[]{}'';:><+=-_&^%$#@《》/\\")
        black_words = ['编辑', '人物', '生平', '背景', '死因', '之谜', '简介', '图片', '来源', '记录', '经历', '演艺经历', '参考资料', '版本', '演员表', '简体名', '作品时间', '剧名类型', '个人成就', '角色介绍', '个人资料', '英文名', '参考', '履历', '图示' ,'业务范围', '时刻表', '基本概述']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments = doc
            segments_label = []
            for item in segments:
                if min_length <= len(item) <= max_length and _check_valid(item):
                    segments_label.append(1)
                else:
                    segments_label.append(0)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                seg_labels.append(cache_label)
                segment_ids.append(cache)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 1))
                cache_dids.extend(item_s)
                index += 1
            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        s_phrase_reps, e_phrase_reps, phrase_sources = [], [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            s_phrase_reps.append(s_rep)
            e_phrase_reps.append(e_rep)
            phrase_sources.extend([(s, e, self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        s_phrase_reps = torch.cat(s_phrase_reps)
        e_phrase_reps = torch.cat(e_phrase_reps)
        phrase_reps = torch.cat([s_phrase_reps, e_phrase_reps], dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx).replace('##', '')) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v3(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs, doc_labels = [], []

        for doc in documents:
            # segments = doc
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-')
                segments.append(item)
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]

            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 1))
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)

            for s, e in zip(s_pos, e_pos):
                context_left = max(0, s-16)
                context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                phrase_sources.append((
                        s, 
                        e, 
                        ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>'),
                        context_string
                ))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                ''
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v4(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs, doc_labels = [], []

        for doc in documents:
            segments = doc
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]
            # ipdb.set_trace()

            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    # add non-punc version
                    if self.retriever.bert_tokenizer.decode(dids[len(cache_dids):len(cache_dids) + len(doc[index]) - 1 + 1])[-1] in [')', ']', '}', '>']:
                        phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 2))
                    else:
                        phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 1))
                    
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
            phrase_sources.extend([
                (
                    s, 
                    e, 
                    # ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('@ - @', '-').replace('@, @', ',').replace('@. @', '.')
                    ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1])
                ) for s, e in zip(s_pos, e_pos)
            ])
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        # phrase_reps = torch.cat(phrase_reps)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx)
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v6(self, documents):

        '''dynamic phrase searcher'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []
        for doc in documents[:-1]:
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']
            seg_ids = [torch.LongTensor(
                [self.retriever.bert_tokenizer.cls_token_id] + i + [self.retriever.bert_tokenizer.sep_token_id]) for i in seg_ids if min_length <= len(i) <= max_length]
            docs.extend(seg_ids)

        # process the prefix document
        prefix = documents[-1][0]
        prefix = prefix.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
        tokens = prefix.split()
        segments = []
        for i in range(len(tokens)):
            for j in range(i+1, i+8):
                segments.append(' '.join(tokens[i:j+1]))
        seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']
        seg_ids = [torch.LongTensor([self.retriever.bert_tokenizer.cls_token_id] + i + [self.retriever.bert_tokenizer.sep_token_id]) for i in seg_ids]
        docs.extend(seg_ids)

        docids = pad_sequence(docs, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids, pad_token_idx=self.retriever.bert_tokenizer.pad_token_id)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]
        phrase_reps = hidden_states[:, 0, :]
        phrase_sources = [(-1, -1, ' ' + self.retriever.bert_tokenizer.decode(i[1:l-1]).replace('[UNK]', '<|endoftext|>')) for i, l in zip(docids, vl)]
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx)
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v5_fast(self, documents):
        '''fast dynamic phrase searcher'''
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []
        for doc in documents:
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id], ids]
                else:
                    cache.append(ids)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        counter, begin_rep, end_rep, phrase_sources = 0, [], [], {}
        doc_id = 0
        collected_docids = []
        for doc_idx, (doc_rep, l, doc_id) in enumerate(zip(hidden_states, vl, docids)):
            l = l.item()
            begin_rep.append(self.retriever.s_proj(doc_rep[1:l-1]))
            end_rep.append(self.retriever.e_proj(doc_rep[1:l-1]))
            begin_counter = counter
            collected_docids.append(doc_id[1:l-1])
            for idx, id_ in enumerate(doc_id[1:l-1]):
                phrase_sources[counter] = (doc_idx, idx, begin_counter + idx, begin_counter + l - 2, '')    # remove the [CLS] and [SEP] tokens
                counter += 1
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        print(f'[!] collect {len(begin_rep)} phrases')

        # packup with the token embeddings
        begin_rep = torch.cat([
            begin_rep,
            self.retriever.token_embeddings[:, :768//2]
        ], dim=0)
        end_rep = torch.cat([
            end_rep,
            self.retriever.token_embeddings[:, 768//2:]
        ], dim=0)
        for idx in range(len(self.retriever.tokenizer)):
            phrase_sources[counter] = (-1, -1, -1, -1, ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx))
            counter += 1
        print(f'[!] add vocabulary and collect {len(begin_rep)} phrases')
        return begin_rep, end_rep, phrase_sources, collected_docids

    @torch.no_grad()
    def process_documents_en_v5(self, documents):

        '''dynamic phrase searcher'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []
        for dd_index, doc in enumerate(documents):
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            moving_pointer, cache_delta_length = 0, 0
            while moving_pointer < len(seg_ids):
                ids = seg_ids[moving_pointer][cache_delta_length:]
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    delta_length = self.args['doc_max_length'] - sum([len(i) for i in cache]) - 2
                    cache_delta_length += delta_length
                    cache.append(ids[:delta_length])
                    if dd_index == 0:
                        cache.append([self.retriever.bert_tokenizer.cls_token_id])
                    else:
                        cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id]]
                else:
                    cache.append(ids)
                    moving_pointer += 1
                    cache_delta_length = 0
            if cache:
                if dd_index == 0:
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                else:
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        for idx, (doc_rep, l, doc_id) in enumerate(zip(hidden_states, vl, docids)):
            s_pos, e_pos = [], []
            if idx == 0:
                first_num = 0
            for i in range(1, l-1-self.args['left_window_size']):
                if self.retriever.bert_tokenizer.decode(doc_id[i]) in self.punc_set:
                    continue
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    # sss = self.retriever.bert_tokenizer.decode(doc_id[j]).replace('##', '')
                    s_pos.append(i)
                    e_pos.append(j)
                    # if sss in self.punc_set:
                    #     break
                if idx == 0:
                    first_num += 1
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>')
                if '<|endoftext|>' in string:
                    continue
                if idx == 0:
                    context_string = ''
                else:
                    context_left = max(0, s-16)
                    context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                if '##' not in string:
                    phrase_sources.append((s, e, string, context_string))
                    # phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.e_proj(e_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        # multiple the weight
        # phrase_reps[:first_num, :] *= 1.1
        # phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        # without the <unk> token
        #non_unk_token_mask = torch.arange(len(self.retriever.tokenizer))
        # non_unk_token_mask = non_unk_token_mask != self.retriever.tokenizer.eos_token_id
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings[:self.retriever.tokenizer.eos_token_id]
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                ''
            ) for idx in range(len(self.retriever.tokenizer)) if idx != self.retriever.tokenizer.eos_token_id
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_v4(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            for w in black_words:
                if w in string:
                    return False
            return True

        def _check_chinese(string, ratio=0.99):
            counter = 0
            for _char in string:
                if '\u4e00' <= _char <= '\u9fa5' or _char in characters:
                    counter += 1
            return counter/len(string) >= ratio

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!?`[]{}'';:><+=-_&^%$#@《》/\\\"")
        black_words = ['[UNK]', '编辑', '人物', '生平', '背景', '死因', '之谜', '简介', '图片', '来源', '记录', '经历', '演艺经历', '参考资料', '版本', '演员表', '简体名', '作品时间', '剧名类型', '个人成就', '角色介绍', '个人资料', '英文名', '参考', '履历', '图示' ,'业务范围', '时刻表', '基本概述']

        # collect candidate phrases
        docs = []
        for doc in documents:
            segments = doc
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id], ids]
                else:
                    cache.append(ids)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)

            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        for doc_rep, l, doc_id in zip(hidden_states, vl, docids):
            s_pos, e_pos = [], []
            for i in range(1, l-1-self.args['left_window_size']):
                if _check_chinese(self.retriever.bert_tokenizer.decode(doc_id[i])) is False:
                    continue
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    if self.retriever.bert_tokenizer.decode(doc_id[j]) in characters:
                        break
                    s_pos.append(i)
                    e_pos.append(j)
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ''.join(self.retriever.bert_tokenizer.convert_ids_to_tokens(doc_id[s:e+1]))
                if _check_valid(string):
                    phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.e_proj(e_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        # phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            # F.normalize(self.retriever.token_embeddings, dim=-1)
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources


    @torch.no_grad()
    def process_documents_v5(self, documents):

        '''dynamic search'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            for w in black_words:
                if w in string:
                    return False
            return True

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?`[]{}'';:><+=-_&^%$#@《》/\\")
        black_words = ['编辑', '人物', '生平', '背景', '死因', '之谜', '简介', '图片', '来源', '记录', '经历', '演艺经历', '参考资料', '版本', '演员表', '简体名', '作品时间', '剧名类型', '个人成就', '角色介绍', '个人资料', '英文名', '参考', '履历', '图示' ,'业务范围', '时刻表', '基本概述']

        # collect candidate phrases
        docs, doc_labels = [], []
        for doc in documents:
            segments = doc
            segments_label = []
            for item in segments:
                if min_length <= len(item) <= max_length and _check_valid(item):
                    segments_label.append(1)
                else:
                    segments_label.append(0)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                seg_labels.append(cache_label)
                segment_ids.append(cache)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, l, doc_id in zip(hidden_states, vl, docids):
            s_pos, e_pos = [], []
            for i in range(1, l-1-self.args['left_window_size']):
                for j in range(
                    min(i+self.args['left_window_size'], l-1),
                    min(i+self.args['right_window_size'], l-1)
                ):
                    s_pos.append(i)
                    e_pos.append(j)
                    if self.retriever.bert_tokenizer.decode(doc_id[j]) in self.punc_set:
                        # punc out of the loop
                        break
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ''.join(self.retriever.bert_tokenizer.convert_ids_to_tokens(doc_id[s:e+1]))
                if '##' not in string:
                    phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.e_proj(e_rep))
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')
        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([(-1, -1, self.retriever.tokenizer.decode(idx)) for idx in range(len(self.retriever.tokenizer))])
        return phrase_reps, phrase_sources


    @torch.no_grad()
    def process_documents_en_v6(self, documents):

        '''dynamic phrase searcher'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []

        for doc in documents:
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id], ids]
                else:
                    cache.append(ids)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        opt_begin_rep, opt_end_rep = [], []
        phrase_sources = []
        for doc_rep, l, doc_id in zip(hidden_states, vl, docids):
            s_pos, e_pos = [], []
            for i in range(1, l-1-self.args['left_window_size']):
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    s_pos.append(i)
                    e_pos.append(j)
                    if self.retriever.bert_tokenizer.decode(doc_id[j]) in self.punc_set:
                        # punc out of the loop
                        break
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>')
                # context_left = max(0, s-16)
                # context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                if '##' not in string:
                    # phrase_sources.append((s, e, string, context_string))
                    phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.input_s_proj(s_rep))
            end_rep.append(self.retriever.input_e_proj(e_rep))
            opt_begin_rep.append(self.retriever.output_s_proj(s_rep))
            opt_end_rep.append(self.retriever.output_e_proj(e_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        opt_begin_rep = torch.cat(opt_begin_rep)
        opt_end_rep = torch.cat(opt_end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        output_phrase_reps = torch.cat([opt_begin_rep, opt_end_rep])
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v5_gpt2(self, documents):

        '''dynamic phrase searcher'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []

        for doc in documents:
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], []
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    segment_ids.append(cache)
                    cache = [ids]
                else:
                    cache.append(ids)
            if cache:
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            dids = list(reversed(dids))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        for doc_rep, l, doc_id in zip(hidden_states, vl, docids):
            s_pos, e_pos = [], []
            for i in range(l-1, self.args['left_window_size'], -1):
                for j in range(
                    max(i-self.args['left_window_size'], 0), 
                    max(i-self.args['right_window_size'], 0)
                ):
                    s_pos.append(i)
                    e_pos.append(j)
                    if self.retriever.bert_tokenizer.decode(doc_id[j]) in self.punc_set:
                        # punc out of the loop
                        break
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.retriever.bert_tokenizer.decode(
                    list(reversed(doc_id[s:e+1]))
                ).replace('[UNK]', '<|endoftext|>')
                if '##' not in string:
                    phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.s_proj_minus(s_rep) - self.retriever.e_proj(e_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_v6(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            for w in black_words:
                if w in string:
                    return False
            return True

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?`[]{}'';:><+=-_&^%$#@《》/\\")
        black_words = ['编辑', '人物', '生平', '背景', '死因', '之谜', '简介', '图片', '来源', '记录', '经历', '演艺经历', '参考资料', '版本', '演员表', '简体名', '作品时间', '剧名类型', '个人成就', '角色介绍', '个人资料', '英文名', '参考', '履历', '图示' ,'业务范围', '时刻表', '基本概述']
    
        # collect candidate phrases
        docs = []
        for dd_index, doc in enumerate(documents):
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(doc, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    if dd_index == 0:
                        cache.append([self.retriever.bert_tokenizer.sep_token_id])
                        # cache.append([self.retriever.bert_tokenizer.cls_token_id])
                    else:
                        cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id], ids]
                else:
                    cache.append(ids)
            if cache:
                if dd_index == 0:
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    # cache.append([self.retriever.bert_tokenizer.cls_token_id])
                else:
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for ddindex, doc in enumerate(docs):
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        for idx, (doc_rep, l, doc_id) in enumerate(zip(hidden_states, vl, docids)):
            s_pos, e_pos = [], []
            if idx == 0:
                first_num = 0
            for i in range(1, l-1-self.args['left_window_size']):
                if self.retriever.bert_tokenizer.decode(doc_id[i]) in self.punc_set:
                    continue
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    sss = self.retriever.bert_tokenizer.decode(doc_id[j]).replace('##', '')
                    s_pos.append(i)
                    e_pos.append(j)
                    if sss in self.punc_set:
                        # punc out of the loop
                        break
                if idx == 0:
                    first_num += 1
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = self.retriever.bert_tokenizer.decode(doc_id[s:e+1])
                context_left = max(0, s-16)
                context_string = self.retriever.bert_tokenizer.decode(doc_id[context_left:s])
                if '##' not in string:
                    phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.e_proj(e_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        '''
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                self.retriever.tokenizer.decode(idx).replace('##', ''),
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        '''
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v7(self, documents):

        '''dynamic phrase searcher with document [CLS] rep'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []
        for dd_index, doc in enumerate(documents):
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            for ids in seg_ids:
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id], ids]
                else:
                    cache.append(ids)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep, document_rep = [], [], []
        phrase_sources = []
        for idx, (doc_rep, l, doc_id) in enumerate(zip(hidden_states, vl, docids)):
            s_pos, e_pos = [], []
            if idx == 0:
                first_num = 0
            for i in range(1, l-1-self.args['left_window_size']):
                if self.retriever.bert_tokenizer.decode(doc_id[i]) in self.punc_set:
                    continue
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    sss = self.retriever.bert_tokenizer.decode(doc_id[j]).replace('##', '')
                    s_pos.append(i)
                    e_pos.append(j)
                    if sss in self.punc_set:
                        break
                if idx == 0:
                    first_num += 1
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>')
                context_left = max(0, s-16)
                context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                if '##' not in string:
                    phrase_sources.append((s, e, string, context_string))
                    # phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            d_rep = doc_rep[0, :].unsqueeze(0).expand(len(new_s_pos), -1)
            begin_rep.append(self.retriever.s_proj(s_rep))
            end_rep.append(self.retriever.e_proj(e_rep))
            document_rep.append(self.retriever.d_proj(d_rep))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        document_rep = torch.cat(document_rep)
        phrase_reps = torch.cat([begin_rep, end_rep, document_rep], dim=-1)
        # multiple the weight
        # phrase_reps[:first_num, :] *= 1.1
        # phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
            # F.normalize(self.retriever.token_embeddings, dim=-1)
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                ''
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def retrieval_generation_beam_search(self, batch):
        '''beam search + copy from the retrieved documents'''
        self.retriever.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        head_weight, tail_weight = batch['head_weight'], batch['tail_weight']
        alpha = batch['coarse_score_alpha']
        beta = 1 - alpha
        coarse_score_softmax_temp = batch['coarse_score_softmax_temp']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        num_beams = batch['beam_search_size']
        _, prefix_length = ids.size()
        beam_size = batch['beam_search_size']

        # init the phrases
        if batch['use_phrase_cache'] is False:
            phrase_reps, phrase_sources = self.process_documents(doc)
            self.cache_phrase_reps, self.cache_phrase_sources = phrase_reps, phrase_sources
        else:
            phrase_reps, phrase_sources = self.cache_phrase_reps, self.cache_phrase_sources
            print(f'[!] load {len(phrase_reps)} cached phrase to save the process time')

        batch_size, seqlen = ids.size()
        generated = []

        ## ========= necessary parameters for beam search ========== #
        # expand the ids to hold num_beams sequences
        assert batch_size == 1
        ids = [ids.squeeze(0).cpu()] * beam_size
        sum_logprobs = torch.zeros(beam_size).cuda()    # [B]
        is_done = [0] * seqlen
        is_first_step = True

        while sum(is_done) < beam_size:
            try:
                query = self.retriever.get_query_rep_beam(ids)    # [B, E] with left padding
                candidates = self.search_from_documents_beam(
                    query, 
                    phrase_reps, 
                    phrase_sources, 
                    search_topk=beam_size, 
                    head_weight=head_weight, 
                    tail_weight=tail_weight
                )
            except:
                ipdb.set_trace()

            # [B*B]
            try:
                candidate_prob = torch.tensor(
                    list(chain(*[
                        [item[1] for item in candidate] for candidate in candidates
                    ]))
                ).cuda()
                candidate_prob = candidate_prob.view(beam_size, beam_size)    # [CB, B]
                candidate_scores = sum_logprobs.unsqueeze(1).expand(-1, beam_size) + candidate_prob    # [B, B]
                candidate_scores = candidate_scores.view(-1)    # [B*B]
                if is_first_step is True:
                    sum_logprobs, index = candidate_scores[:beam_size].topk(beam_size, dim=-1, largest=True, sorted=True)
                else:
                    sum_logprobs, index = candidate_scores.topk(beam_size, dim=-1, largest=True, sorted=True)
            except:
                ipdb.set_trace()

            try:
                # collect the topk-index ids and append to the ids
                topk_ids, beam_lengths = [], []
                for index_ in index.tolist():
                    beam_id = index_ // beam_size
                    token_id = index_ % beam_size
                    candidate = candidates[beam_id][token_id]
                    sub_ids = self.retriever.tokenizer.encode(candidate[0], add_special_tokens=False)
                    sub_ids = torch.LongTensor(sub_ids)
                    ids_ = torch.cat((ids[beam_id], sub_ids), dim=-1)
                    topk_ids.append(ids_)
                is_done = [len(i) >= seqlen + self.test_max_len for i in topk_ids]
                ids = topk_ids
                is_first_step = False
            except:
                ipdb.set_trace()
        generated = self.retriever.tokenizer.decode(ids[0][seqlen:seqlen+self.test_max_len])
        return generated, 0.

    @torch.no_grad()
    def process_documents_en_v8(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs, doc_labels = [], []

        for doc in documents:
            # segments = doc
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-')
                segments.append(item)
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]

            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 1))
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)

            for s, e in zip(s_pos, e_pos):
                context_left = max(0, s-16)
                context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                phrase_sources.append((
                        s, 
                        e, 
                        ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>'),
                        context_string
                ))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                ''
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources
    
    @torch.no_grad()
    def process_documents_en_v9(self, documents):

        '''dynamic phrase searcher'''

        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs = []
        for dd_index, doc in enumerate(documents):
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-').replace('< unk >', '[UNK]')
                segments.append(item)
            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, cache = [], [[self.retriever.bert_tokenizer.cls_token_id]]
            moving_pointer, cache_delta_length = 0, 0
            while moving_pointer < len(seg_ids):
                ids = seg_ids[moving_pointer][cache_delta_length:]
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    delta_length = self.args['doc_max_length'] - sum([len(i) for i in cache]) - 2
                    cache_delta_length += delta_length
                    cache.append(ids[:delta_length])
                    if dd_index == 0:
                        cache.append([self.retriever.bert_tokenizer.cls_token_id])
                    else:
                        cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    segment_ids.append(cache)
                    cache = [[self.retriever.bert_tokenizer.cls_token_id]]
                else:
                    cache.append(ids)
                    moving_pointer += 1
                    cache_delta_length = 0
            if cache:
                if dd_index == 0:
                    cache.append([self.retriever.bert_tokenizer.cls_token_id])
                else:
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                segment_ids.append(cache)
            docs.extend(segment_ids)

        # collect the phrases
        docids = []
        for doc in docs:
            dids = list(chain(*doc))
            docids.append(torch.LongTensor(dids))
        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)
        vl = docids_mask.sum(dim=-1)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        for idx, (doc_rep, l, doc_id) in enumerate(zip(hidden_states, vl, docids)):
            s_pos, e_pos = [], []
            if idx == 0:
                first_num = 0
            for i in range(1, l-1-self.args['left_window_size']):
                # if self.retriever.bert_tokenizer.decode(doc_id[i]) in self.punc_set:
                #     continue

                # 不输出 <unk> 相关的内容
                if self.retriever.bert_tokenizer.decode(doc_id[i]) in ['<', '>', 'unk']:
                    continue

                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    sss = self.retriever.bert_tokenizer.decode(doc_id[j])
                    # 不输出 <unk> 相关的内容
                    if sss in ['<', '>', 'unk']:
                        break
                    
                    if sss.startswith('##'):
                        # remove the last and append the new
                        if s_pos and e_pos:
                            s_pos.pop()
                            e_pos.pop()
                    s_pos.append(i)
                    e_pos.append(j)
                    sss = sss.replace('##', '')
                    
                last_index = min(i+self.args['right_window_size'], l-1)
                sss = self.retriever.bert_tokenizer.decode(doc_id[last_index])
                if sss.startswith('##'):
                    if s_pos and e_pos:
                       s_pos.pop()
                       e_pos.pop()
                
                if idx == 0:
                    first_num += 1
            new_s_pos, new_e_pos = [], []
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>')
                if '<|endoftext|>' in string:
                    continue
                if idx == 0:
                    context_string = ''
                else:
                    context_left = 0
                    context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                if '##' not in string:
                    phrase_sources.append((s, e, string, context_string))
                    # phrase_sources.append((s, e, string))
                    new_s_pos.append(s)
                    new_e_pos.append(e)
            s_rep = doc_rep[new_s_pos, :]
            e_rep = doc_rep[new_e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        # multiple the weight
        # phrase_reps[:first_num, :] *= 1.1
        # phrase_reps = F.normalize(phrase_reps, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        # without the <unk> token
        # non_unk_token_mask = torch.arange(len(self.retriever.tokenizer))
        # non_unk_token_mask = non_unk_token_mask != self.retriever.tokenizer.eos_token_id
        phrase_reps = torch.cat([
            phrase_reps,
            # self.retriever.token_embeddings[:self.retriever.tokenizer.eos_token_id]
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                'TOKEN'
            # ) for idx in range(len(self.retriever.tokenizer)) if idx != self.retriever.tokenizer.eos_token_id
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    @torch.no_grad()
    def process_documents_en_v10(self, documents):
        self.retriever.eval()
        min_length, max_length = self.args['min_phrase_length'], self.args['max_phrase_length']
        # collect candidate phrases
        docs, doc_labels = [], []

        for doc in documents:
            # segments = doc
            segments = []
            for item in doc:
                item = item.replace('<unk>', '[UNK]').replace('@,@', ',').replace('@.@', '.').replace('@-@', '-')
                segments.append(item)
            segments_label = [1 if min_length <= len(item) <= max_length else 0 for item in segments]

            seg_ids = self.retriever.bert_tokenizer.batch_encode_plus(segments, add_special_tokens=False)['input_ids']

            # split the subchunk by the length
            segment_ids, seg_labels, cache, cache_label = [], [], [[self.retriever.bert_tokenizer.cls_token_id]], [0]
            for label, ids in zip(segments_label, seg_ids):
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['doc_max_length']:   # [CLS] and [SEP] tokens
                    cache.append([self.retriever.bert_tokenizer.sep_token_id])
                    cache_label.append(0)
                    segment_ids.append(cache)
                    seg_labels.append(cache_label)
                    cache, cache_label = [[self.retriever.bert_tokenizer.cls_token_id], ids], [0, label]
                else:
                    cache.append(ids)
                    cache_label.append(label)
            if cache:
                cache.append([self.retriever.bert_tokenizer.sep_token_id])
                cache_label.append(0)
                segment_ids.append(cache)
                seg_labels.append(cache_label)

            docs.extend(segment_ids)
            doc_labels.extend(seg_labels)

        # collect the phrases
        docids, phrase_positions = [], []
        for doc, label in zip(docs, doc_labels):
            phrases = []
            dids = list(chain(*doc))
            cache_dids = []
            index = 0
            for item_s, item_l in zip(doc, label):
                if item_l == 1:
                    phrases.append((len(cache_dids), len(cache_dids) + len(doc[index]) - 1))
                cache_dids.extend(item_s)
                index += 1

            docids.append(torch.LongTensor(dids))
            phrase_positions.append(phrases)

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.bert_tokenizer.pad_token_id)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        begin_rep, end_rep = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)

            for s, e in zip(s_pos, e_pos):
                context_left = max(0, s-16)
                context_string = ' ' + self.retriever.bert_tokenizer.decode(doc_id[context_left:s]).replace('[UNK]', '<|endoftext|>')
                phrase_sources.append((
                        s, 
                        e, 
                        ' ' + self.retriever.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>'),
                        context_string
                ))
            
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.retriever.s_proj(begin_rep), self.retriever.e_proj(end_rep)], dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')

        # packup with the token embeddings
        phrase_reps = torch.cat([
            phrase_reps,
            self.retriever.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.retriever.tokenizer.decode(idx) if self.retriever.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.retriever.tokenizer.decode(idx),
                ''
            ) for idx in range(len(self.retriever.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources


