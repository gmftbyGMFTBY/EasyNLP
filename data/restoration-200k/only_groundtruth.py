with open('train.txt') as f:
    dataset = [line for line in f.readlines() if line[0]== '1']

with open('train_only_groundtruth.txt', 'w') as f:
    for line in dataset:
        f.write(line)
