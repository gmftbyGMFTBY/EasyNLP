# The experiments of the Comparison-based Dialog Response Selection (DRS)

replace the NSP (next sentence prediction) with the CCE (candidate comparison evaluation) learning object.

## 1. Traditional Settings

In traditional settings, the comparison-based DRS model use the fully comparison to generate the scores for ranking.

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| BERT-CCE           |       |       |       |       |

### 1.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | **51.4**  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| BERT-CCE           | 32.79 | 53.46 | 87.24 | 68.62 | 51.72 | 64.67 |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| BERT-CCE       |       |       |       |       |

### 1.4 Restoration-200k Dataset (Optional)

| Models             | R10@1 | R10@2 | R10@5 | MRR   | P@1 | MAP |
| ------------------ | ----- | ----- | ----- | ----- | --- | --- |
| dual-bert          | | | | | | |
| bert-ft            | | | | | | |
| compare-encoder    | 54.76   | 73.19      |  92.51     | 74.99      | 61.37 | 73.63 |

## 2. Two-stage Boosting for Existing DRS models

### 2.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| UMS-BERT           |       |       |       |       |
| UMS-BERT+CCE       |       |       |       |       |
| BERT-FP            |       |       |       |       |
| BERT-FP+CCE        |       |       |       |       |
| dual-bert          |       |       |       |       |
| dual-bert+CCE      |       |       |       |       |

### 2.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| BERT-FP+CCE        | 33.0  | 53.56 | **86.65**  | 68.72  | 52.17  | **64.41**   |
| UMS-BERT           |       |       |       |       |       |        |
| UMS-BERT+CCE       |       |       |       |       |       |        |
| dual-bert          |       |       |       |       |       |        |
| dual-bert+CCE      |       |       |       |       |       |        |

### 2.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| BERT-FP+CCE    | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| UMS-BERT       |       |       |       |       |
| UMS-BERT+CCE   |       |       |       |       |
| dual-bert      |       |       |       |       |
| dual-bert+CCE  |       |       |       |       |
