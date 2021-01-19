from datetime import datetime
from tqdm import tqdm
import ipdb
import torch
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import argparse
import random
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
from dataset import BertPostTrainingDataset
from pretrained_dpt import BertNSPMLM, BertMLM, BertNSP


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="ecommerce")
    arg_parser.add_argument("--data_path", type=str, required=True)
    arg_parser.add_argument("--bert_pretrained", type=str, default="bert-base-uncased")
    arg_parser.add_argument("--save_path", type=str)
    arg_parser.add_argument('--local_rank', type=int)
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--lr', type=float, default=3e-5)
    arg_parser.add_argument('--epoch', type=int, default=2)
    arg_parser.add_argument('--seed', type=float, default=50)
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1)
    arg_parser.add_argument('--amp_level', type=str, default='O2')
    arg_parser.add_argument('--grad_clip', type=float, default=5.0)
    arg_parser.add_argument('--model', type=str, default='nspmlm')
    return arg_parser.parse_args()


class PostTraining:
    
    def __init__(self, args):
        self.args = args
        self.recoder = SummaryWriter(self.args['save_path'])
        if self.args['model'] == 'nspmlm':
            print(f'[!] Loss contains MLM and NSP')
        elif self.args['model'] == 'mlm':
            print(f'[!] Loss only contains MLM')
        else:
            print(f'[!] Wrong model: {self.args["model"]}')
        
    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
        
    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

    def _build_dataloader(self):
        self.train_dataset = BertPostTrainingDataset(self.args['data_path'], mode="train")
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args['batch_size'],
            shuffle=False,
        )
        print("[!] Dataloader Finish")

    def _build_model(self):
        if self.args['model'] == 'nspmlm':
            self.model = BertNSPMLM(self.args)
        elif self.args['model'] == 'mlm':
            self.model = BertMLM(self.args)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=self.args['amp_level']
        )
        self.model = nn.parallel.DistributedDataParallel(
            self.model, 
            device_ids=[self.args['local_rank']],
            output_device=self.args['local_rank'],
        )
        print(" [!] Building Model Finished")

    def train(self):
        self._build_dataloader()
        self._build_model()

        accu_loss, accu_count = 0, 0
        for epoch in tqdm(range(self.args['epoch'])):
            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                buffer_batch = batch.copy()
                if torch.cuda.is_available():
                    for key in batch:
                        # ipdb.set_trace()
                        buffer_batch[key] = buffer_batch[key].cuda()

                self.optimizer.zero_grad()
                
                loss = self.model(buffer_batch)
                accu_loss += loss.item()
                accu_count += 1

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
                
                self.optimizer.step()

                description = f"[Epoch: {epoch}][Loss: {round((accu_loss / accu_count), 4)}]"
                tqdm_batch_iterator.set_description(description)
                
                # tensorboard
                self.recoder.add_scalar(f'train-epoch/RunLoss', loss.item(), batch_idx)
                self.recoder.add_scalar(f'train-epoch/Loss', accu_loss/accu_count, batch_idx)

            if self.args['local_rank'] == 0:
                self.save_model(self.args['save_path'])
                print(f'[!] save model into {self.args["save_path"]}')

if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    print('[!] parameters:')
    print(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
                                         
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model = PostTraining(args)
    model.train()