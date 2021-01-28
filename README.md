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

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder     | 80.6  | 90.6  | 98.3  | 88.01  |
<!--| Bi-Encoder-CL  | 78.9  | 89.8  | 98.0  | 86.95  |-->
| Bi-Encoder-CL2 | 80.6  | 91.1  | 98.2  | 88.16  |
<!--| Bi-Encoder-VAE(Mean/1) | 80.6  | 92.2  | 97.9  | 88.38  |
| Bi-Encoder-VAE(Mean/5) | 80.4  | 92.2  | 98.0  | 88.27  |
| Bi-Encoder-VAE(Mean/10) | 80.2  | 92.3  | 97.9  | 88.18  |
| Bi-Encoder-VAE(Mean/20) | 80.2  | 92.3  | 98.0  | 88.19  |
| Bi-Encoder-VAE(Max/1)  | 80.6  | 92.2  | 97.9  | 88.38  |
| Bi-Encoder-VAE(Max/2)  | 80.5  | 92.3  | 98.0  | 88.34  |
| Bi-Encoder-VAE(Max/3)  | 80.1  | 92.2  | 97.8  | 88.11  |
| Bi-Encoder-VAE(Max/4)  | 80.6  | 92.3  | 97.9  | 88.4   |-->
| Bi-Encoder-VAE(Max/5)  | 80.7  | 92.4  | 97.9  | 88.43  |
<!--| Bi-Encoder-VAE(Max/6)  | 80.2  | 92.0  | 97.9  | 88.15  |
| Bi-Encoder-VAE(Max/7)  | 80.2  | 92.1  | 97.8  | 88.14  |
| Bi-Encoder-VAE(Max/8)  | 80.3  | 92.2  | 98.1  | 88.22  |
| Bi-Encoder-VAE(Max/9)  | 80.4  | 92.2  | 97.8  | 88.26  |
| Bi-Encoder-VAE(Max/10) | 80.0  | 91.9  | 97.8  | 88.00  |
| Bi-Encoder-VAE2(Max/2) | 79.5  | 90.2  | 97.8  | 87.30  |
| Bi-Encoder-VAE2(Max/5) | 79.3  | 90.3  | 97.6  | 87.20  |-->
| Bi-Encoder-VAE2(Max/10)| 79.7  | 89.8  | 97.8  | 87.34  |
| Poly-Encoder   | 80.0  | 90.3  | 97.9  | 87.69  |
| BERT-FT        | 62.3  | 84.2  | 98    | 77.59  |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |
| BERT-Gen-FT    | 63.3  | 83.5  | 97.1  | 77.71  |
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
| SOTA           | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5  |
| Bi-Encoder     | 28.57 | 47.46 | 81.35 | 63.33 | 45.28 | 59.6  |
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
