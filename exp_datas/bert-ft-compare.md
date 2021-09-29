# BERT-FT-Compare Model for Dialog Response Selection and Automatic Evaluation Metric

## 1. Dialog Response Selection

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| BERT-FT-Compare    |       |       |       |       |

### 1.2 Douban Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | **51.4**  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| BERT-FT-Compare    |       |       |       |       |      |     |

### 1.3 Ubuntu-v1 Dataset

| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| BERT-FT-Compare|       |       |       |       |      |     |

### 1.4 HORSe Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  | 14800.94  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  | 15135.32  |
| SA-BERT+BERT-FP    | 51.46 | 69.43  | 92.82 | 71.99 | 57.07 | 70.72  | 15135.32  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| DR-BERT(full)      | 56.44 | 74.7  | 93.56 | 75.91 | 62.42 | 74.75  | 21150.52  |
| BERT-FT-Compare    | 63.16 | 80.39 | 95.93 | 80.95 | 69.49 | 79.82  |

## 2. Automatic Evaluation Metric
