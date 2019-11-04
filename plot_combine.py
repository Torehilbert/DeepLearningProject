import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


folder = r"C:\Users\ToreH\OneDrive - Københavns Universitet\Skole\02456 Deep Learning\Project\DeepLearningProject\output"

all_tags = []
all_bs = []
all_lr = []
all_rp = []

contents = os.listdir(folder)
for cont in contents:
    splits = cont.split('_')
    tag = splits[0]
    all_tags.append(tag)

    bs = splits[1]
    all_bs.append(bs)

    lr = splits[2]
    all_lr.append(lr)

    rp = splits[3]
    all_rp.append(rp)

bs_unq = np.unique(all_bs)
lr_unq = np.unique(all_lr)
rp_unq = np.unique(all_rp)

base_score = np.zeros(shape=(len(bs_unq), len(lr_unq)))

for cont in contents:
    print(cont)
    splits = cont.split('_')
    if(splits[0]=='trainR'):
        bs_indx = np.where(splits[1] == bs_unq)[0]
        lr_indx = np.where(splits[2] == lr_unq)[0]
        df = pd.read_csv(os.path.join(folder, cont))
        base_score[bs_indx, lr_indx] += np.sum(200 - df.values)/df.values.shape[0]

base_score = base_score/len(rp_unq)

result_path = os.path.join(folder, 'result.csv')
f = open(result_path, 'w')

for c in range(base_score.shape[1]):
    f.write('lr=%s' % lr_unq[c])
    if(c != base_score.shape[1]-1):
        f.write(',')
    else:
        f.write('\n')

for r in range(base_score.shape[0]):
    f.write('bs=%s' % bs_unq[r])
    f.write(',')
    for c in range(base_score.shape[1]):
        f.write(str(base_score[r, c]))
        if(c != base_score.shape[1]-1):
            f.write(',')
        else:
            f.write('\n')
f.close()
