from model.utils import *
from .gpt2_original import GPT2OriginalModel
from model.RepresentationModels import DensePhraseEncoder
from .utils import *
from config import *

class CopyGenerationEncoder(nn.Module):

    def __init__(self, **args):
        super(CopyGenerationEncoder, self).__init__()
        self.args = args
        model_name = args['pretrained_model']

        retriever_args = deepcopy(self.args)
        retriever_args['model'] = 'phrase-copy'
        config = load_config(retriever_args)
        retriever_args.update(config)
        
        generator_args = deepcopy(self.args)
        generator_args['model'] = 'gpt2-original'
        config = load_config(generator_args)
        generator_args.update(config)

        self.retriever = DensePhraseEncoder(**retriever_args) 
        self.generator = GPT2OriginalModel(**generator_args)
        self.criterion_ = nn.CrossEntropyLoss(ignore_index=self.generator.vocab.pad_token_id, reduction='none')
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.generator.vocab.pad_token_id)

        # texsmart engine
        # self.engine = NluEngine('/home/johntianlan/sources/texsmart-sdk-0.3.0-m-zh/data/nlu/kb', 1)

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
    def search_from_documents(self, query, phrase_reps, phrase_source):
        self.retriever.eval()
        self.generator.eval()
        dp = torch.matmul(query, phrase_reps.t()).squeeze(0)   
        topk = dp.topk(self.args['search_topk'], dim=-1)[1]    # [K]
        candidates = [''.join(phrase_source[i][-1].split()) for i in topk]
        return candidates

    def search_from_faiss(self, query):
        candidates = self.faiss_searcher._search(query.cpu().numpy(), topk=self.args['recall_topk'])[0]
        return candidates

    def chinese_tokenization(self, doc, seg_url="http://100.77.13.7:8082"):
        req = {
            "str": doc,
            "options": {
                "pos_tagging":{"enable":False,"alg":"dnn"},
                "ner": {
                    "enable": False,
                    "alg": "fine.high_acc"
                }
            }
        }
        result = requests.post(seg_url, data=json.dumps(req))
        word_list = json.loads(result.content)["phrase_list"]
        seg_sentence = []
        for each_word in word_list:
            seg_sentence.append(each_word["str"])
        return seg_sentence
 
    @torch.no_grad()
    def process_documents(self, documents):
        self.retriever.eval()
        self.generator.eval()

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

            # texsmart segmentation
            segments = list(jieba.cut(doc))
            # segments = texsmart_segmentation(self.engine, doc)
            # segments = self.chinese_tokenization(doc)

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
                if sum([len(i) for i in cache]) + len(ids) + 2 > self.args['max_doc_length']:   # [CLS] and [SEP] tokens
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
                        if min_length <= len(p_cache_dids) - 1 - b_ + 1 <= max_length:
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
            phrase_sources.extend([(s, e, self.retriever.bert_tokenizer.decode(doc_id[s:e])) for s, e in zip(s_pos, e_pos)])
        phrase_reps = torch.cat(phrase_reps)
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
    def rerank(self, input_ids, candidates):
        self.retriever.eval()
        self.generator.eval()

        ids, labels = [], []
        cids, cpos = [], []
        for candidate in candidates: 
            candidate = self.generator.vocab.encode(candidate, add_special_tokens=False)
            ids_, candidate_ = self.truncation(input_ids.tolist(), candidate, self.args['max_rerank_len'])
            ids.append(ids_ + candidate_[:-1])
            labels.append([self.generator.pad] * (len(ids_) - 1) + candidate_) 
            cids.append(ids_ + candidate_)
            cpos.append((len(ids_), len(cids[-1])))

        ids = [torch.LongTensor(i) for i in ids]
        cids = [torch.LongTensor(i) for i in cids]
        labels = [torch.LongTensor(i) for i in labels]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.generator.pad)
        cids = pad_sequence(cids, batch_first=True, padding_value=self.generator.pad)
        ods = pad_sequence(labels, batch_first=True, padding_value=self.generator.pad)
        ids_mask = generate_mask(ids)
        cids_mask = generate_mask(cids)
        ids, ods, ids_mask, cids, cids_mask = to_cuda(ids, ods, ids_mask, cids, cids_mask)

        # ppl scores
        ppl_scores = self.calculate_ppl(ids, ids_mask, ods)

        # contrastive scores
        contrastive_scores = self.calculate_contrastive(cids, cids_mask, cpos)
        scores = np.array(ppl_scores) + np.array(contrastive_scores) * self.args['contrastive_phrase_score_penalty']

        # phrases = [(candidate, score) for candidate, score in zip(candidates, scores)]
        phrases = [(candidate, score) for candidate, score in zip(candidates, scores)]
        phrases = sorted(phrases, key=lambda x:x[1])
        return phrases[0][0]

    @torch.no_grad()
    def retrieval_generation_search_e2e(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        self.generator.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        ids = batch['ids']
        batch_size, seqlen = ids.size()
        generated = []
        while len(ids[0]) < seqlen + self.generator.test_max_len:
            # init the query
            query = self.retriever.get_query_rep(ids)
            # search candidate phrases
            candidates = self.search_from_faiss(query)
            ipdb.set_trace()
            break
            # rerank candidates by ppl
            candidate = self.rerank(ids[0], candidates)
            # phrase ppl
            candidate_ids = torch.LongTensor(
                self.generator.vocab.encode(candidate, add_special_tokens=False)
            ).cuda().unsqueeze(0)
            candidate_length = len(candidate_ids[0])
            phrase_ids = torch.cat([ids, candidate_ids], dim=-1)
            ods = phrase_ids.clone().detach()
            ods[:, :-candidate_length] = self.generator.vocab.pad_token_id
            phrase_ids = phrase_ids[:, :-1]
            ods = ods[:, 1:]
            candidate_ppl = self.calculate_ppl_v2(phrase_ids, ods)
            # token ppl
            # cs_ids = ids.clone().detach()
            # for _ in range(candidate_length):
            cs_ids = self.decoding_one_step(ids, generation_method, topk=topk, topp=topp, beam_width=beam_width, model_prediction_confidence=model_prediction_confidence)
            cs_ods = cs_ids.clone().detach()
            cs_ods[:, :-candidate_length] = self.generator.vocab.pad_token_id
            cs_ids_ = cs_ids.clone().detach()[:, :-1]
            cs_ods = cs_ods[:, 1:]
            token_ppl = self.calculate_ppl_v2(cs_ids_, cs_ods)
            # dynamci switching
            if candidate_ppl < phrase_alpha * token_ppl:
                ids = torch.cat([ids, candidate_ids], dim=-1)
                generated.append(f' [{candidate}] ')
            else:
                ids = cs_ids
                generated.append(self.generator.vocab.convert_ids_to_tokens(ids[0, -1].item()))
        return ''.join(generated)
    
    @torch.no_grad()
    def retrieval_generation_search(self, batch):
        '''contrastive search + copy from the retrieved documents'''
        self.retriever.eval()
        self.generator.eval()
        generation_method = batch['generation_method']
        topk, topp, beam_width, model_prediction_confidence = batch['topk'], batch['topp'], batch['beam_width'], batch['model_prediction_confidence']
        phrase_alpha = batch['phrase_alpha']
        update_step = batch['update_step']
        ids = batch['ids']
        doc = batch['docs']    # textual documents
        # init the phrases
        phrase_reps, phrase_sources = self.process_documents(doc)
        batch_size, seqlen = ids.size()
        generated = []
        while len(ids[0]) < seqlen + self.generator.test_max_len:
            # init the query
            query = self.retriever.get_query_rep(ids)
            # search candidate phrases
            candidates = self.search_from_documents(query, phrase_reps, phrase_sources)
            ipdb.set_trace()
            break
            # rerank candidates by ppl
            candidate = self.rerank(ids[0], candidates)
            # phrase ppl
            candidate_ids = torch.LongTensor(
                self.generator.vocab.encode(candidate, add_special_tokens=False)
            ).cuda().unsqueeze(0)
            candidate_length = len(candidate_ids[0])
            phrase_ids = torch.cat([ids, candidate_ids], dim=-1)
            ods = phrase_ids.clone().detach()
            ods[:, :-candidate_length] = self.generator.vocab.pad_token_id
            phrase_ids = phrase_ids[:, :-1]
            ods = ods[:, 1:]
            candidate_ppl = self.calculate_ppl_v2(phrase_ids, ods)
            # token ppl
            # cs_ids = ids.clone().detach()
            # for _ in range(candidate_length):
            cs_ids = self.decoding_one_step(ids, generation_method, topk=topk, topp=topp, beam_width=beam_width, model_prediction_confidence=model_prediction_confidence)
            cs_ods = cs_ids.clone().detach()
            cs_ods[:, :-candidate_length] = self.generator.vocab.pad_token_id
            cs_ids_ = cs_ids.clone().detach()[:, :-1]
            cs_ods = cs_ods[:, 1:]
            token_ppl = self.calculate_ppl_v2(cs_ids_, cs_ods)
            # dynamci switching
            if candidate_ppl < phrase_alpha * token_ppl:
                ids = torch.cat([ids, candidate_ids], dim=-1)
                generated.append(f' [{candidate}] ')
            else:
                ids = cs_ids
                generated.append(self.generator.vocab.convert_ids_to_tokens(ids[0, -1].item()))
            # dynamic update the documents
            if (len(ids[0]) - seqlen) % update_step == 0:
                string = ''.join(self.generator.vocab.convert_ids_to_tokens(ids[0]))
                doc = self.retrieve_doc(string, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
                phrase_reps, phrase_sources = self.process_documents(doc)
        return ''.join(generated)

    def decoding_one_step(self, ids, generation_method, topk=1., topp=1., beam_width=5, model_prediction_confidence=0.4):
        if generation_method == 'contrastive-search':
            ids = ContrastiveDecodingOneStep(
                self.generator.model, 
                ids, 
                beam_width, 
                model_prediction_confidence, 
                self.generator.unk,
            )
        elif generation_method == 'greedy-search':
            logits = self.generator.model(ids).logits[-1, -1, :]
            logits[self.generator.unk] = -np.inf
            next_token = logits.max(dim=-1)[1]
            ids = torch.cat((ids, next_token.reshape(1, 1)), dim=1)
        elif generation_method == 'topk-topp-search':
            logits = self.generator.model(ids).logits[-1, -1, :]
            logits[self.generator.unk] = -np.inf
            filtered_logits = top_k_top_p_filtering(logits, top_k=topk, top_p=topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=-1)
        else:
            raise Exception(f'[!] Unknow generation method: {generation_method}')
        return ids

    def calculate_ppl(self, input_ids, ids_mask, labels):
        outputs = self.generator.model(input_ids=input_ids, attention_mask=ids_mask)
        logits = outputs.logits
        bsz, seqlen = input_ids.size()
        mle_loss = self.criterion_(logits.view(-1, self.generator.vocab_size), labels.view(-1)).reshape(bsz, seqlen)
        losses = []
        for mle_loss_, mask in zip(mle_loss, labels):
            loss = mle_loss_[mask != self.generator.pad].mean().exp().item()
            losses.append(loss)
        return losses

    def calculate_ppl_v2(self, input_ids, labels):
        logits = self.generator.model(input_ids=input_ids).logits
        mle_loss = self.criterion(logits.view(-1, self.generator.vocab_size), labels.view(-1))
        return math.exp(mle_loss.item())

    def calculate_contrastive(self, ids, ids_mask, pos):
        # labels: [B, S]
        output = self.generator.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-1]   # [B, S, E]
        hidden_states = F.normalize(hidden_states, dim=-1)    # [B, S, E]
        scores = []
        for hidden_state, (pos_s, pos_e) in zip(hidden_states, pos):
            context_hidden_state = hidden_state[:pos_s]    # [S_1, E]
            target_hidden_state = hidden_state[pos_s:pos_e]    # [S_2, E]
            s = torch.matmul(context_hidden_state, target_hidden_state.t()).t()    # [S_2, S_1]
            s = s.max(dim=-1)[0].mean(dim=-1)    # [S_1]
            scores.append(s.item())
        return scores

    @torch.no_grad()
    def work(self, data):

        '''generation_method: topk-topp, greedy search, contrastive search, pure retrieval, retrieval+generation'''

        decoding_method = data['decoding_method'] 
        generation_method = data['generation_method']
        topk, topp, beam_width, model_prediction_confidence = data['topk'], data['topp'], data['beam_width'], data['model_prediction_confidence']
        phrase_alpha = data['phrase_alpha']
        update_step = data['update_step']
        assert generation_method in ['contrastive-search', 'greedy-search', 'topk-topp-search']
        prefix = data['prefix']
        ground_truth = data['ground_truth']

        ids = self.generator.vocab.encode(prefix, add_special_tokens=False)
        ids = torch.LongTensor(ids).unsqueeze(0).cuda()
        batch = {
            'ids': ids,        
        }

        if decoding_method == 'topk-topp-search':
            response = self.generator.topk_topp_search(batch)
        elif decoding_method == 'greedy-search':
            response = self.generator.greedy_search(batch)
        elif decoding_method == 'contrastive-search':
            response = self.generator.contrastive_search(batch)
        elif decoding_method == 'beam-search':
            response = self.generator.beam_search(batch)
        elif decoding_method == 'retrieval-search':
            docs = self.retrieve_doc(prefix, recall_topk=self.args['recall_topk'], max_query_len=self.args['max_query_len'])
            ids = self.retriever.tokenizer.encode(prefix, add_special_tokens=False)
            ids = torch.LongTensor(ids).unsqueeze(0).cuda()
            batch = {
                'ids': ids, 
                'docs': docs, 
                'phrase_alpha': np.inf, 
                'generation_method': generation_method,
                'topk': topk, 
                'topp': topp,
                'beam_width': beam_width,
                'model_prediction_confidence': model_prediction_confidence,
                'update_step': update_step,
                'prefix_text': prefix,
            }
            response = self.retrieval_generation_search(batch)
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
