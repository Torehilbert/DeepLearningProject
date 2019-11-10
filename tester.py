import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def probarray(prob):
    return torch.FloatTensor([prob, (1-prob)/4, (1-prob)/4, (1-prob)/4, (1-prob)/4])

def entropy(probs):
    return -(probs.log() * probs).sum()


pvals = np.linspace(0.0001, 0.9999, 500)
es = np.zeros(shape=pvals.shape)
for i in range(len(pvals)):
    p = pvals[i]
    A = probarray(p)
    e = entropy(A)
    es[i] = e.numpy()

plt.figure()
plt.plot(pvals, -es)
plt.show()



# actions = []
# for i in range(10000):
#     action = (random.random() < np.cumsum(np.squeeze(A.detach().numpy()))).argmax()
#     actions.append(action)

# print(np.unique(actions, return_counts=True))
