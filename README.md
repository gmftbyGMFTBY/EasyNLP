## Cross-encoder models performance

Parameters reference: [TODO](https://github.com/taesunwhang/UMS-ResSel/blob/635e37f5340faf5a37f3b1510a9402be18348c66/config/hparams.py)

## How to Use

1. create post data from the orignial dataset

```bash
cd data;
python create_post_data.py
```

2. create data for post training

```bash
# dataset saved in data/ecommerce/train_post.hdf5
./data/create_post_train_dataset.sh ecommerce
```

3. post train

```bash
# checkpoint are saved in ckpt/ecommerce/bert-post/*
./post_train/post-train.sh ecommerce <nspmlm/mlm/nsp> <gpu_ids>
```

4. load post train checkpoint and fine-tuning on response selection corpus

```bash
./run.sh train-post ecommerce bert-ft <gpu_ids>
```

5. inference on the train dataset, save in FAISS index

```bash
# save the extended train dataset into data/<dataset>/candidates.pt
./run.sh inference ecommerce dual-bert <gpu_ids>
```

## Ready models and datasets

### 1. Models
1. BERT-ft
2. BERT-ft-gen
3. Bi-Encoder
4. Poly-Encoder


### 2. Datasets
1. E-Commerce
2. Douban
3. Ubuntu-v1


## Experiment Results
### 1. E-Commerce Dataset

_Note:_
* Bi-Encoder and Poly-Encoder use in-batch negative approach during training procedure, and the number of negative samples is equal to the bsz (default 16) minus 1.
* batch size is 16
* for Bi-Encoder-VAE, max strategy is better than mean
* sequence length is 256
* Compared with Bi-Encoder-CL, Bi-Encoder-CL2 fully leverage the variant response represeantion
* more negative samples, better performance
* google position embedding is better than absolute position embedding
* max strategy is equal to mean strategy

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder(bsz=16) | 79.7  | 90.8  | 98.3  | 87.63  |
| Bi-Encoder-one2many(bsz=16,head=5,mean) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-one2many(bsz=16,head=5,max) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-hier(bsz=16) | 80.5  | 91.1  | 98.5  | 88.14  |
| Bi-Encoder-hier-multi(bsz=16,m=5) | 80.6  | 91.6  | 98.6  | 88.01  |
| Bi-Encoder(bsz=64) | 83.7  | 92.4  | 98.5  | 90.02  |
| Bi-Encoder-CL2 | 80.6  | 91.1  | 98.2  | 88.16  |
| Bi-Encoder-VAE(Max/5)  | 80.7  | 92.4  | 97.9  | 88.43  |
| Bi-Encoder-VAE2(Max/5) | 80.2  | 91.9  | 98.8  | 88.11  |
| Bi-Encoder-VAE2(Max/10) | 80.4  | 92.0  | 98.7  | 88.22  |
| Poly-Encoder   | 80.0  | 90.3  | 97.9  | 87.69  |
| BERT-FT        | 62.3  | 84.2  | 98    | 77.59  |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |
| BERT-Gen-FT    | 63.3  | 83.5  | 97.1  | 77.71  |
| BERT-Gen-FT w/o Gen | | | | |

### 2. Douban Dataset

* one2many performance is worse than dual-bert on douban corpus, which is very different from the ecommerce corpus. The reason maybe: the dual-bert performance is worse than that on ecommerce corpus, which lead to bad candidate samples for dual-bert-one2many model, e.g., the quality of the candidates matter!
* SOLUTION1: separate the head, reduce the side effect of the bad candidates
* SOLUTION2: improve the coarse retrieved candidates quality
* SOLUTION3: MoE (Mixture of the Experts)?

| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| Bi-Encoder(bsz=16) | 28.16 | 48.5  | 80.87 | 63.64 | 46.18 | 59.38  |
| Bi-Encoder(bsz=16,head=5,max) | 28.32 | 46.64  | 81.32 | 62.83 | 44.98 | 58.74  |
| Bi-Encoder(bsz=60) | 30.24 | 50.32 | 83.09 | 65.33 | 47.98 | 61.38  |
| BERT-FT        | 25.86 | 44.63 | 83.43 | 61.55 | 42.58 | 57.59 |
| BERT-FT+MLM+NSP|       |       |       |       |       |       |
| BERT-FT+MLM    |       |       |       |       |       |       |
| BERT-FT+NSP    |       |       |       |       |       |       |
| BERT-Gen-FT    |       |       |       |       |       |       |
| BERT-Gen-FT w/o Gen |      |      |      |      |     |     |

### 3. Ubuntu Dataset

* batch size: 48
* max sequence length: 256

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 88.4  | 94.6  | 99.0  | 97.5   |
| Bi-Encoder(bsz=48) | 67.22  | 80.23      | 94.6      |        |
| BERT-FT        |       |       |       |        |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |
