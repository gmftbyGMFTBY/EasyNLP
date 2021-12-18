from model.utils import *
from inference_utils import *

class GPT2MemoryEncoder(nn.Module):

    '''gpt2 model with additional phrase memory'''

    def __init__(self, **args):
        super(GPT2MemoryEncoder, self).__init__()
        model = args['pretrained_model']
        self.vocab = BertTokenizer.from_pretrained(model)

        # special tokens
        self.pad = self.vocab.pad_token_id
        self.unk = self.vocab.unk_token_id
        self.cls = self.vocab.cls_token_id
        self.sep = self.vocab.sep_token_id
        self.special_tokens = set([self.pad, self.unk, self.cls, self.sep])
        self.test_max_len = args['test_max_len']
        self.args = args

        # model

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        self.model.eval()
        return self.model.calculate_ppl(ids, ids_mask, label)

    @torch.no_grad()
    def predict(self, batch):
        '''greedy search with batch inference, pad in the left'''
        self.model.eval()
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        ids_pos = batch['pos_ids']
        batch_size, seqlen = ids.size()
        generated = [[] for _ in range(batch_size)]
        past_key_values = None
        while True:
            next_token, past_key_values = self.model.predict_one_step(
                ids,
                ids_mask,
                ids_pos,
                past_key_values,
            )
            for idx, t in enumerate(next_token.tolist()):
                generated[idx].append(t)
            if max([len(i) for i in generated]) > self.test_max_len:
                break
            # reconstruct the ids and ids_mask
            ids = next_token.unsqueeze(1)    # [B, 1]
            ids_mask = torch.ones_like(ids)    # [B, 1]
            ids_pos = 1 + ids_pos[:, -1].unsqueeze(dim=-1)
        # remove the special tokens
        rest = []
        for g in generated:
            g = [i for i in g if i not in self.special_tokens]
            rest.append(g)
        return rest

    @torch.no_grad()
    def build_offline_index(self, dataloader, output_file):
        self.model.eval()
        embd, text, effective_index = [], [], []
        pbar = tqdm(dataloader)
        for batch in pbar:
            if batch is None:
                continue
            e, t, i = self.model.offline_inference(batch)
            embd.append(e)
            text.extend(t)
            effective_index.extend(i)
            pbar.set_description(f'[!] got phrase: {len(effective_index)}')
        embd = torch.cat(embd).cpu().numpy()    # [S, E]
        torch.save(
            (embd, text, effective_index), 
            f'{output_file}_{self.args["local_rank"]}',
        )

        # only the first process could build the faiss index
        torch.distributed.barrier()
        # load all of the embeddings
        embds, texts, effective_indexs = [], [], []
        for i in tqdm(range(torch.distributed.get_world_size())):
            embd, text, effective_index = torch.load(
                f'{output_file}_{i}'        
            )
            embds.append(embd)
            texts.extend(text)
            effective_indexs.extend(effective_index)
        embds = np.concatenate(embds)
        assert len(embds) == len(text) == len(effective_index)
        self.model.cached_index._build(
            embds, rest_texts, rest_eintexts, speedup=True
        )
        self.model.cached_index.save(
            self.model.faiss_path, self.model.corpus_path
        )
        return len(embds)
    
    def forward(self, batch):
        gpt2_ids, gpt2_ids_mask = batch['ids'], batch['ids_mask']
        loss_token, loss_phrase, token_acc , phrase_acc = self.model(gpt2_ids, gpt2_ids_mask)
        return loss_token, loss_phrase, token_acc, phrase_acc
