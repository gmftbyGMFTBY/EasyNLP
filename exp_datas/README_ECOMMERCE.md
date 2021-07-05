# E-Commerce Dataset

* recall performance

| Model (CPU/109105)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat(q-r)    |  0.567 | 0.791   | 29.51               |
| dual-bert-fusion(q-r)  |  0.561 | 0.645   | -              |

* rerank performance

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| BERT-FP        | 87.0  | 95.6  | 99.3  | -      |
| dual-bert      | 91.7  | 96.0  | 99.2  | 94.85  |
| bert-ft-compare    | 94.2  | 97.8  | 99.5  | 96.52  |
| bert-ft-compare(epoch=10, super_hard_negative) | 93.1 | 98.5 | 99.9 | 96.22 |
| dual-bert          | 91.7  | 96.0  | 99.2  | 94.85  |
| dual-bert+compare  | 92.0  | 96.6  | 99.6  | 95.23  |
| bert-ft            | 83.4  | 94.4  | 99.4  | 90.43  |
| bert-ft+compare(with not sure)    | 93.7  | 98.0  | 99.6  | 96.34  |
| bert-ft+compare(without not sure)    | 92.7  | 97.3  | 99.6  | 95.72  |
| dual-bert(bsz=16, epoch=5, bert-post) | 86.1 | 94.1 | 99.2 | 91.71 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c) | 88.6 | 94.5 | 99.1 | 93.04 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c/c-c/r-r) | 87.2 | 94.8 | 99.3 | 92.37 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c, single) | 85.4 | 94.0 | 99.0 | 91.25 |
