## Cross-encoder models performance

Parameters reference: [TODO](https://github.com/taesunwhang/UMS-ResSel/blob/635e37f5340faf5a37f3b1510a9402be18348c66/config/hparams.py)

## How to Use

1. create data for post training

```bash
# dataset saved in data/ecommerce/train_post.hdf5
./data/create_post_train_dataset.sh ecommerce
```

2. post train

```bash
# checkpoint are saved in ckpt/ecommerce/bert-post/*
./post_train/post-train.sh ecommerce <nspmlm/mlm/nsp> <gpu_ids>
```

3. load post train checkpoint and fine-tuning on response selection corpus

```bash
./run.sh train-post ecommerce bert-ft <gpu_ids>
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

| Original       | R10@1 | R10@2 | R10@5 | MRR   |
| -------------- | ----- | ----- | ----- | ----- |
| Bi-Encoder     |       |       |       |       |
| Poly-Encoder   |       |       |       |       |
| BERT-FT        | 62.3  | 84.2  | 98    | 77.59 |
| BERT-FT+MLM+NSP|       |       |       |       |
| BERT-FT+MLM    |       |       |       |       |
| BERT-FT+NSP    |       |       |       |       |
| BERT-Gen-FT    | 63.3  | 83.5  | 97.1  | 77.71 |
| BERT-Gen-FT w/o Gen | | | | |

<!--
| Adversarial   | R10@1 | R10@2 | R10@5 | MRR    |
| ------------- | ----- | ----- | ----- | ------ |
| BERT-FT       | 37.4  | 73.4  | 97.6  | 62.84  |
| BERT-Gen-FT   | 44.1  | 74.8  | 96.1  | 66.23  |
| BERT-Gen-FT w/o Gen | | | | |-->

### 2. Douban Dataset

| Original       | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP  |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| Bi-Encoder     |       |       |       |       |       |       |
| BERT-FT        | 25.86 | 44.63 | 83.43 | 61.55 | 42.58 | 57.59 |
| BERT-FT+MLM+NSP|       |       |       |       |       |       |
| BERT-FT+MLM    |       |       |       |       |       |       |
| BERT-FT+NSP    |       |       |       |       |       |       |
| BERT-Gen-FT    |       |       |       |       |       |       |
| BERT-Gen-FT w/o Gen |      |      |      |      |     |     |

<!--
| Adversarial   | R10@1 | R10@2 | R10@5 | MRR    |  P@1  | MAP  |
| ------------- | ----- | ----- | ----- | ------ | ----- | ---- |
| BERT-FT       |  |  |  |  |       |      |
| BERT-Gen-FT   |  |  |  |  |       |      |
| BERT-Gen-FT w/o Gen | | | | |-->