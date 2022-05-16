from model.utils import *
from .gpt2_original import GPT2OriginalModel
from model.RepresentationModels import DensePhraseEncoder, DensePhraseV2Encoder, DensePhraseV3Encoder, DensePhraseV4Encoder
from .utils import *
from config import *

class CopyGenerationEncoder(nn.Module):

    def __init__(self, **args):
        super(CopyGenerationEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        generator_args = deepcopy(self.args)
        generator_args['model'] = 'gpt2-original'
        config = load_config(generator_args)
        generator_args.update(config)
        self.generator = GPT2OriginalModel(**generator_args) 

        retriever_args = deepcopy(self.args)
        retriever_args['model'] = 'phrase-copy'
        config = load_config(retriever_args)
        retriever_args.update(config)
        self.retriever = DensePhraseV4Encoder(**retriever_args) 
        self.test_max_len = self.args['test_max_len']

    def init_searcher(self, agent, searcher, base_data):
        self.search_agent = agent
        self.searcher = searcher  
        self.base_data = base_data
        print(f'[!] init the simcse search agent over')

    def init_faiss_searcher(self, searcher):
        self.faiss_searcher = searcher
        print(f'[!] init the faiss searcher agent over')

    def search_docs(self, prefix):
        batch = self.searcher_agent.inference_context_one_sample(prefix).cpu().numpy()
        retrieval_list = self.searcher._search(batch, topk=self.args['recall_topk'])[0]
        docs = [test_data.base_data[key] for jey in retrieval_list]
        return docs

    @torch.no_grad()
    def search_from_words(self, query):
        self.retriever.eval()
        dp = torch.matmul(query, self.retriever.token_embeddings.t()).squeeze(0)
        dis, topk = dp.topk(self.args['search_topk'], dim=-1)
        dis = dis.tolist()
        topk = topk.tolist()
        candidates = [(self.retriever.tokenizer.convert_ids_to_tokens(i), round(d, 4)) for i, d in zip(topk, dis)]
        return candidates

    @torch.no_grad()
    def search_from_documents(self, query, phrase_reps, phrase_source):
        self.retriever.eval()
        dp = torch.matmul(query, phrase_reps.t()).squeeze(0)   
        dis, topk = dp.topk(self.args['search_topk'], dim=-1)    # [K]
        dis = dis.tolist()
        candidates = [(''.join(phrase_source[i][-1].split()), round(d, 4)) for i, d in zip(topk, dis)]
        return candidates

    def search_from_faiss(self, query):
        candidates, distance = self.faiss_searcher._search_dis(query.cpu().numpy(), topk=500)
        candidates, distance = candidates[0], [i.item() for i in distance[0]]
        new_candidates = []
        new_candidates_pool = set()
        for c, d in zip(candidates, distance):
            if c in new_candidates_pool:
                continue
            new_candidates_pool.add(c)
            new_candidates.append((c, d))
        return new_candidates[:self.args['recall_topk']]

    @torch.no_grad()
    def process_documents(self, documents):
        '''TODO: add the sentence as the phrase set'''
        self.retriever.eval()

        def _check_valid(string):
            for char in string:
                if char in characters:
                    return False
            return True

        # init
        characters = set(".,，。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?`[]{}'';:><+=-_&^%$#@《》/\\")
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

        docids = pad_sequence(docids, batch_first=True, padding_value=self.retriever.pad)
        docids_mask = generate_mask(docids)
        docids, docids_mask = to_cuda(docids, docids_mask)

        output = self.retriever.phrase_encoder(docids, docids_mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][-1]    # [B, S, E]

        phrase_reps, phrase_sources = [], []
        for doc_rep, doc_pos, doc_id in zip(hidden_states, phrase_positions, docids):
            s_pos, e_pos = [i for i, j in doc_pos], [j for i, j in doc_pos]
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            rep = torch.cat([s_rep, e_rep], dim=-1)
            phrase_reps.append(rep)
            phrase_sources.extend([(s, e, self.retriever.bert_tokenizer.decode(doc_id[s:e+1])) for s, e in zip(s_pos, e_pos)])
        phrase_reps = torch.cat(phrase_reps)
        hidden_states = F.normalize(hidden_states, dim=-1)
        assert len(phrase_reps) == len(phrase_sources)
        print(f'[!] collect {len(phrase_reps)} phrases')
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
        update_step = batch['update_step']
        ids = batch['ids']
        batch_size, seqlen = ids.size()
        generated = []
        while len(ids[0]) < seqlen + self.test_max_len:
            # init the query
            query = self.retriever.get_query_rep(ids)
            # search candidate phrases
            candidates = self.search_from_faiss(query)
            return candidates
        return ''.join(generated)

    @torch.no_grad()
    def retrieval_generation_search_one_step(self, batch):
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
        query = self.retriever.get_query_rep(ids)
        phrase_candidates = self.search_from_documents(query, phrase_reps, phrase_sources)
        word_candidates = self.search_from_words(query)
        phrase_candidates = [list(i) + ['phrase'] for i in phrase_candidates]
        word_candidates = [list(i) + ['word'] for i in word_candidates]
        candidates = [c for c in phrase_candidates + word_candidates if '[UNK]' not in c[0]]
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
        return candidates
    
    @torch.no_grad()
    def retrieval_generation_search(self, batch):
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
        while len(ids[0]) < seqlen + self.test_max_len:
            query = self.retriever.get_query_rep(ids)
            phrase_candidates = self.search_from_documents(query, phrase_reps, phrase_sources)
            word_candidates = self.search_from_words(query)

            phrase_candidates = [list(i) + ['phrase'] for i in phrase_candidates]
            word_candidates = [list(i) + ['word'] for i in word_candidates]

            candidates = [c for c in phrase_candidates + word_candidates if '[UNK]' not in c[0]]
            # candidates = [c for c in word_candidates if '[UNK]' not in c[0]]
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            candidate_tokens = [item[0] for item in candidates]
            candidate_prob = torch.tensor([item[1] for item in candidates]).cuda()

            ids, candidate = self.decoding_one_step_inner(ids, candidate_tokens, candidate_prob, generation_method, topk=topk, topp=topp, model_prediction_confidence=model_prediction_confidence)
            generated.append(f'{candidate} ')
            
            if (len(ids[0]) - seqlen) % update_step == 0:
                string = ''.join(self.retriever.tokenizer.convert_ids_to_tokens(ids[0]))
                doc = self.retrieve_doc(string, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
                phrase_reps, phrase_sources = self.process_documents(doc)
        generated = ''.join(generated)
        return generated

    def decoding_one_step_inner(self, ids, candidates, candidates_prob, generation_method, topk=1., topp=1., model_prediction_confidence=0.4):
        if generation_method == 'contrastive-search':
            candidates_ = self.retriever.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            ids, index = ContrastiveDecodingOneStepUnify(
                self.retriever.model, 
                ids, 
                candidates_,
                candidates_prob,
                model_prediction_confidence, 
                self.retriever.pad,
                self.args['temperature']
            )
            candidate = candidates[index]
        elif generation_method == 'greedy-search':
            candidate = candidates[0]
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
        return ids, candidate

    @torch.no_grad()
    def work(self, data):

        '''generation_method: nucleus-search, greedy-search, contrastive-search'''

        decoding_method = data['decoding_method'] 
        generation_method = data['generation_method']
        topk, topp, beam_width, model_prediction_confidence = data['topk'], data['topp'], data['beam_width'], data['model_prediction_confidence']
        phrase_alpha = data['phrase_alpha']
        update_step = data['update_step']
        assert generation_method in ['contrastive-search', 'greedy-search', 'nucleus-search']
        prefix = data['prefix']
        ground_truth = data['ground_truth']
        debug_index = data['debug_index']

        ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
        ids = torch.LongTensor(ids).unsqueeze(0).cuda()
        batch = {
            'ids': ids,        
        }

        if decoding_method == 'topk-topp-search':
            response = self.generator.topk_topp_search(batch)
            pass
        elif decoding_method == 'greedy-search':
            response = self.generator.greedy_search(batch)
            pass
        elif decoding_method == 'contrastive-search':
            response = self.generator.contrastive_search(batch)
            pass
        elif decoding_method == 'beam-search':
            response = self.generator.beam_search(batch)
            pass
        elif decoding_method == 'word-nucleus-search':
            batch['test_max_len'] = self.test_max_len
            batch['topk'] = topk
            batch['topp'] = topp
            response = self.retriever.nucleus_search(batch)
        elif decoding_method == 'word-greedy-search':
            batch['test_max_len'] = self.test_max_len
            response = self.retriever.greedy_search(batch)
        elif decoding_method == 'word-contrastive-search':
            batch['test_max_len'] = self.test_max_len
            batch['beam_width'] = beam_width
            batch['model_prediction_confidence'] = model_prediction_confidence
            response = self.retriever.contrastive_search(batch)
        elif decoding_method == 'retrieval-generation-search-onestep':
            docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
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
        elif decoding_method == 'retrieval-generation-search':
            docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
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
            response = self.retrieval_generation_search(batch)
        elif decoding_method == 'retrieval-generation-search-e2e':
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
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
        return response

    def retrieve_doc(self, string, recall_topk=50, max_query_len=512):
        rep = self.search_agent.inference_context_one_sample(string, max_len=max_query_len).cpu().numpy()
        doc_list = self.searcher._search(rep, topk=recall_topk)[0]
        docs = [self.base_data[k] for k in doc_list]
        return docs
