from model.utils import *
from dataloader.util_func import *

class BERTEssayRetrieval(nn.Module):

    '''Bert-based essay evaluation with multi auxiliary tasks:
        1. deletion detection: [DEL]
        2. insert detection: [INS]
            * random
            * related
        3. order restoration: [ORD]
        4. masked language model
        5. auto-regression model
        6. next session prediction
    '''

    def __init__(self, **args):
        super(BERTEssayRetrieval, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']
        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertModel.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.vocab = BertTokenizer.from_pretrained(model)
        self.vocab_size = len(self.vocab) + 3    # [INS], [DEL], [SRH]

        # deleteion head
        self.deletion_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 2)
        )
        self.deletion_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # insertion head
        self.insertion_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Liner(768, 2)
        )
        self.insertion_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # search head
        self.search_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Liner(768, 2)
        )
        self.search_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # mlm head
        self.mlm_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, self.vocab_size)
        )
        self.mlm_critieron = nn.CrossEntropyLoss(ignore_index=-1)

        # clm head
        self.clm_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, self.vocab_size)
        )
        self.clm_critieron = nn.CrossEntropyLoss(ignore_index=-1)

        # nsp (next session prediction) head
        self.nsp_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.nsp_criterion = nn.CrossEntropyLoss()
        
        # npp (next passage prediction) head
        self.npp_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.npp_criterion = nn.CrossEntropyLoss()

        self.map = {
            'del': (self.deletion_head, self.deletion_criterion),  
            'ins': (self.insertion_head, self.insertion_criterion),  
            'srh': (self.search_head, self.search_criterion),  
            'mlm': (self.mlm_head, self.mlm_criterion),  
            'clm': (self.clm_head, self.clm_criterion),  
            'nsp': (self.nsp_head, self.nsp_criterion),  
            'npp': (self.npp_head, self.npp_criterion),  
        }

    def forward(self, batch):
        ids, mask = batch['ids'], batch['attn_mask']
        label = batch['label']    # [B, S] / [B]
        logits = self.model(ids, mask)[0]    # [B, S, E]
        # repackup
        packages = {t: [[], []] for t in ['del', 'ins', 'srh', 'mlm', 'clm', 'nsp', 'npp']}
        for idx, t in enumerate(batch['type']):
            # logits
            packages[t][0].append(logits[idx])
            # label
            packages[t][1].append(label[idx])
        # get loss
        loss = 0
        for t in ['del', 'ins', 'srh', 'mlm', 'clm', 'nsp', 'npp']:
            logits, labels = packages[t]
            if len(logits) == 0:
                continue
            logits = torch.stack(logits)
            label = torch.stack(labels)
            head, criterion = self.map[t]
            logits = head(logits)
            loss += criterion(logits, label)
        return loss
