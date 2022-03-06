from model.utils import *


class BERTFTHierEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTFTHierEncoder, self).__init__()
        model = args['pretrained_model']
        self.model = BertEmbedding(model=model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['trs_hidden_size'], 
            nhead=args['trs_nhead']
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=args['trs_nlayer'])
        self.proj = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.gray_cand_num = args['gray_cand_num']

    def _encode(self, ids, ids_mask, turn_length):
        reps = self.model(ids, ids_mask)
        reps = torch.split(reps, turn_length)
        return reps

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])
        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        for cid_rep in cid_reps:
            m = torch.tensor([False] * len(cid_rep) + [True] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        cid_mask = torch.stack(cid_mask).cuda()    # [B, S]

        reps = self.fusion_layer(
            reps.permute(1, 0, 2),
            src_key_padding_mask=cid_mask,
        ).permute(1, 0, 2)    # [B, S, E]
        reps = reps[range(len(reps)), 0, :]    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = cid.shape[0]
        reps = self._encode(cid, cid_mask, turn_length)
        reps = self.get_context_level_rep(reps, turn_length)
        reps = self.proj(reps)[:, 1]
        return reps

    def forward(self, batch):
        cid = batch['ids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']

        batch_size = cid.shape[0]
        reps = self._encode(cid, cid_mask, turn_length)

        # prepare more random negative samples
        responses = [items[-1] for items in reps]
        new_reps, labels = [], []
        for idx, items in enumerate(reps):
            # positive samples
            new_reps.append(items)
            # negative samples
            random_index = list(range(len(reps)))
            random_index.remove(idx)
            random_index = random.sample(random_index, self.gray_cand_num)
            for ridx in random_index:
                new_reps.append(torch.cat([items[:-1], responses[ridx].unsqueeze(0)], dim=0))
            labels.extend([1] + [0] * self.gray_cand_num)
        # random shuffle
        random_index = list(range(len(labels)))
        random.shuffle(random_index)
        new_reps = [new_reps[i] for i in random_index]
        labels = torch.LongTensor([labels[i] for i in random_index]).cuda()

        reps = self.get_context_level_rep(new_reps, turn_length)
        reps = self.proj(reps)    # [B, 2]
        loss = self.criterion(reps, labels)
        acc = (reps.max(dim=-1)[1] == labels).to(torch.float).mean().item()
        return loss, acc
