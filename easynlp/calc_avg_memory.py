with open('memory_usage_dual_bert_hier.txt') as f:
    data = [int(line.strip()[5:]) for line in f.readlines() if line.strip()]

import numpy as np
print(np.mean(data))
