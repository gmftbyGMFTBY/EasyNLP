# Ubuntu V1 Dataset

* rerank performance

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 0.884 | 0.946 | 0.990 | 0.975  |
| HCL            | 0.867 | 0.940 | 0.992 | 0.977  |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 0.977  |
| dual-bert(bsz=16, epoch=5, bert-post) | 84.69 | 92.66 | 98.51 | - |
| dual-bert(bsz=64, epoch=5, bert-fp) | 87.64 | 94.22 | 98.57  | - |

    1000 test samples for fast iterations

| Model             | R10@1 | R10@2 | R10@5 | MRR    |
| ----------------- | ----- | ----- | ----- | ------ |
| bert-ft-compare(pos=0.15)   | 88.7  | 94.1  | 97.1  | 92.69  |
| bert-ft-compare(pos=0.15, super-hard-negative)   | 89.8  | 95.9  | 99.3  | 93.91  |
| bert-ft           | 89.7  | 96.0  | 99.3  | 93.92  |
| bert-ft+compare(margin=-0.1)   | 87.7  | 94.6  | 99.3  | 92.63  |
| bert-ft+compare(margin=0.0)   | 87.1  | 93.9  | 99.1  | 92.14  |
| bert-ft+compare(margin=0.1)   | 86.2  | 93.6  | 98.9  | 91.59  |
| bert-ft+compare   | 89.1  | 95.1  | 99.0  | 93.38  |
| bert-ft+compare(pos_margin=-0.2)   | 89.0  | 94.9  | 99.0  | 93.3  |
| dual-bert         | 86.8  | 94.5  | 98.5  | 91.98  |
| dual-bert+compare(pos_margin=0.1) | 87.2  | 94.0  | 98.3  | 92.1  |
| dual-bert+compare(pos_margin=-0.1) | 88.2  | 94.6  | 98.3  | 92.72  |
| dual-bert+compare(pos_margin=-0.2) | 87.7  | 94.1  | 98.1  | 92.33  |

* recall performance

    Because of the very large test set, we use the LSH other than the Flat

| Originali (545868)       | Top-20 | Top-100 | Time |
| -------------- | ----- | ----- | ------ |
| dual-bert-LSH | 0.1374 | 0.2565 | 8.59  |
| dual-bert-fusion-LSH | 0.7934 | 0.8147 | 7.89  |
| ES(q-q) | 0.0101 | 0.0202 | 22.13 |
| ES(q-r) | 0.0014 | 0.0083 | 9.79 |