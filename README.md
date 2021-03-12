## State-of-the-art retrieval-based multi-turn response selection baselines

Constrastive Learning for dual-encoder model, leveraging the memory bank to enlarge the number of the negative samples. Besides, the FAISS memory bank can be used to recall better samples, so these better samples can be used as the medium samples (not so good and not so bad).

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
5. Bi-Encoder+one2many


### 2. Datasets
1. E-Commerce
2. Douban
3. Ubuntu-v1
4. LCCC


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
* worse quality (bi-encoder-one2many-bad) of the candidates could bring better performance (test whether the number of the negative samples matter, rather than the quality)
* the number of the samples and the quality of samples matter!!!

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder(bsz=16) | 80.8  | 91.7  | 98.3  | 88.39  |
| Bi-Encoder-mb(bsz=16, mb=4096) | 80.6  | 91.0  | 97.9  | 88.09  |
| Bi-Encoder(bsz=64) | 83.7  | 92.4  | 98.5  | 90.02  |
| Bi-Encoder-one2many(bsz=16) | 90.4  | 95.6  | 98.9  | 94.16  |
| Bi-Encoder-one2many(bsz=16, bert-post, single head) | 91.7 | 95.4  | 99.4  | 94.83  |
| Bi-Encoder-one2many-bad(bsz=16,head=5,max,pre-extract=50) | 91.0  | 95.5  | 99.3  | 94.46  |
| Bi-Encoder-one2many-pseudo(bsz=16,head=5,max,pre-extract=50) | 89.4  | 95.2  | 99.2  | 93.5  |
| Bi-Encoder-one2many-concat(bsz=16,head=5,max,pre-extract=50) | 89.5  | 94.4  | 98.8  | 93.42  |
| Bi-Encoder-one2many-no-additional-loss(bsz=16,head=5,max) | 75.5  | 89.1  | 98.2  | 85.28  |
| Bi-Encoder-one2many(bsz=16,head=5,mean,pre-extract=10) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 86.1  | 92.9  | 98.5  | 91.3  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 88.2  | 94.7  | 99.2  | 92.85  |

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder(bsz=16) | 79.7  | 90.8  | 98.3  | 87.63  |
| Bi-Encoder-one2many(bsz=16,head=5,mean) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-one2many-bad(bsz=16,head=5,max) | 91.0  | 95.5  | 99.3  | 94.46  |
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 86.1  | 92.9  | 98.5  | 91.3  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 88.2  | 94.7  | 99.2  | 92.85  |
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
* max-sequence-length and PLMs (hfl/chinese-bert-wwm is slightly worse than bert-base-chinese) is not the core, but max-sequence-length do improve some metrics
* good candidates (pre-extract=20, topk=10) provide  better R10@1 and R10@2, hopeful!
* SOLUTION1: separate the head, reduce the side effect of the bad candidates
* SOLUTION2: improve the coarse retrieved candidates quality
* SOLUTION3: MoE (Mixture of the Experts)?

| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| Bi-Encoder(max-len=512,bsz=16) | 28.1 | 47.85  | 83.05 | 64.45 | 46.63 | 59.36  |
| Bi-Encoder(bsz=16) | 28.16 | 48.5  | 80.87 | 63.64 | 46.18 | 59.38  |
| Bi-Encoder-one2many(bsz=16, bert-post, single-head) | 26.69 | 50.34  | 82.53 | 63.6 | 44.08 | 59.24  |
| Bi-Encoder-one2many-no-addition-loss(bsz=16, bert-post, single-head) | 28.99 | 50.07  | 84.28 | 64.81 | 46.93 | 60.96  |
| Bi-Encoder(bsz=16, bert-post) | 28.9 | 49.51  | 84.7 | 64.62 | 46.63 | 60.98  |
| Bi-Encoder(chinese-bert-wwm,bsz=16) | 28.14 | 46.89  | 81.92 | 63.19 | 45.43 | 59.11  |
| Bi-Encoder(bsz=60) | 30.24 | 50.32 | 83.09 | 65.33 | 47.98 | 61.38  |
| Bi-Encoder-one2many-pseudo(bsz=16,pre-extract=50) | 23.89 | 42.43 | 79.49 | 57.98 | 37.63 | 55.13 |
| Bi-Encoder-one2many(bsz=16,pre-extract=50) | 26.39 | 47.1 | 81.56 | 61.77 | 42.48 | 58.19 |
| Bi-Encoder-one2many-no-additional-loss(bsz=16,pre-extract=50) | 28.89 | 48.39 | 81.13 | 63.4 | 45.28 | 59.63 |
| Bi-Encoder-one2many-concat(bsz=16,pre-extract=200) | 27.2 | 46.04  | 80.46 | 61.83 | 43.63 | 58.33|
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 22.53 | 39.03  | 77.26 | 55.92 | 34.93 | 52.71  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 28.32 | 46.64  | 81.32 | 62.83 | 44.98 | 58.74  |
| BERT-FT        | 25.86 | 44.63 | 83.43 | 61.55 | 42.58 | 57.59 |
| BERT-FT(bert-post) | 27.51 | 46.67 | 84.39 | 62.92 | 43.93 | 59.46 |
| BERT-FT(bert-post) | | | | | | |
| BERT-FT+MLM+NSP|       |       |       |       |       |       |
| BERT-FT+MLM    |       |       |       |       |       |       |
| BERT-FT+NSP    |       |       |       |       |       |       |

### 3. LCCC Dataset

* max sequence length: 256

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | -     | -     | -     | -      |
| Bi-Encoder(bsz=16) | 21.5  | 31.2      | 47.1   | 36.65   |
| Bi-Encoder-one2many(bsz=16,max,pre-extract=50) | 27.0 | 36.7 | 54.0 | 41.85 |
| BERT-FT        |       |       |       |        |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |

### 4. Ubuntu V1 Dataset

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 0.884 | 0.946 | 0.990 | 0.975  |
| Bi-Encoder(bsz=48) | 67.22 | 80.23     | 94.6 | -  |
| Bi-Encoder-one2many(bsz=16,max,pre-extract=50) |  |      |  |  |
| BERT-FT        | 66.86 | 79.75 | 94.15 | -     |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |

