import numpy as np
import torch
import random
import time


policy_distribution = torch.FloatTensor([[0.1, 0.2, 0.5, 0.2]])

samples = []

t0 = time.time()
for i in range(100000):
    action = (random.random() < np.cumsum(np.squeeze(policy_distribution.detach().numpy()))).argmax()
    samples.append(action)
t1 = time.time()

unqs, counts = np.unique(samples, return_counts=True)
print(counts/np.sum(counts))
print(t1-t0)


rands = np.random.rand(100000)
samples = []

t0 = time.time()
m = torch.distributions.Categorical(probs=policy_distribution)
for i in range(100000):
    action = m.sample().numpy()
    #action = (rands[i] < np.cumsum(np.squeeze(policy_distribution.detach().numpy()))).argmax()
    samples.append(action)


t1 = time.time()

unqs, counts = np.unique(samples, return_counts=True)
print(counts/np.sum(counts))
print(t1 - t0)