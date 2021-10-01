# BERT-FT-Compare Model for Dialog Response Selection and Automatic Evaluation Metric

## 1. Dialog Response Selection

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| BERT-FT-Compare(bert-fp,margin=0.0)    |  85.1     |   94.3    |    99.1   |    91.2    |
| BERT-FT-Compare(bert-fp,margin=0.1)    |  96.0     |   99.6    |    99.8   |    97.89   |
| BERT-FT-Compare(bert-fp,margin=0.2)    |  97.5     |   99.6    |    99.8   |    98.63   |

### 1.2 Douban Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| BERT-FT-Compare    |       |       |       |       |       |        |

### 1.3 Ubuntu-v1 Dataset

| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 97.7   |
| BERT-FT-Compare|       |       |       |        |

### 1.4 HORSe Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  |
| SA-BERT+BERT-FP    | 51.46 | 69.43 | 92.82 | 71.99 | 57.07 | 70.72  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   |
| DR-BERT(full)      | 56.44 | 74.7  | 93.56 | 75.91 | 62.42 | 74.75  |
| BERT-FT-Compare(bert-fp-comp)    | 64.23 | 80.96 | 94.79 | 81.38 | 70.4  | 80.29  |
| BERT-FT-Compare(bert-fp,margin=-0.1)   | 64.39 | 80.65 | 96.26 | 81.76 | 71.01  | 80.75  |
| BERT-FT-Compare(bert-fp,margin=0.0)    | 58.91 | 75.83 | 94.61 | 77.47 | 64.95  | 76.32  |
| BERT-FT-Compare(bert-fp,margin=0.1)    | 86.12 | 95.19 | 98.51 | 96.3 | 93.43  | 95.19  |
| BERT-FT-Compare(bert-fp,margin=0.2)    | 86.83 | 95.57 | 98.87 | 96.81 | 94.24  | 95.66  |

* HORSe ranking test set

| Models           | NDCG@3 | NDCG@5 |
| ---------------- | ------ | ------ |
| BERT             | 0.625  | 0.714  |
| BERT-FP          | 0.609  | 0.709  |
| SA-BERT(BERT-FP) | 0.674  | 0.753  |
| Poly-encoder     | 0.679  | 0.765  |
| DR-BERT(full)    | 0.699  | 0.774  |
| BERT-FT-Compare(bert-fp-comp)  | 0.663  | 0.749  |
| BERT-FT-Compare(bert-fp,margin=0.0)  | 0.679  | 0.758  |
| BERT-FT-Compare(bert-fp,margin=0.1)  | 0.547  | 0.649  |
| BERT-FT-Compare(bert-fp,margin=-0.1) | 0.669  | 0.747  |


## 2. Automatic Evaluation Metric
