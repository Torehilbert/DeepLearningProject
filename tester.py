import numpy as np
import time
import random

def choice(p):
    return (np.random.rand() < np.cumsum(p)).argmax()

def manual_three_choice(r, p):
    if(r < p[0]):
        return 0
    elif(r < p[1]):
        return 1
    else:
        return 2

def choice_dynamic(r, p):
    for i in range(len(p)):
        if(r < p[i]):
            return i
    return len(p)

reps = 10000
N = 15
rand_logits = np.random.rand(N)
rand_sum = np.sum(rand_logits)
p = rand_logits / rand_sum

print(p)
print(rand_sum)

t0 = time.time()
for i in range(reps):
    A = np.random.choice(N, p=p)

t1 = time.time()
for i in range(reps):
    A = (random.random() < np.cumsum(p)).argmax()

t2 = time.time()
for i in range(reps):
    A = (np.random.rand() < np.cumsum(p)).argmax()

t3 = time.time()
prerolls = np.random.rand(reps)
for i in range(reps):
    A = (prerolls[i] < np.cumsum(p)).argmax()
t4 = time.time()

for i in range(reps):
    A = manual_three_choice(prerolls[i], p)
t5 = time.time()

for i in range(reps):
    A = choice_dynamic(prerolls[i], p)
t6 = time.time()


print("M1: %f"%(t1-t0))
print("M2: %f"%(t2-t1))
print("M3: %f"%(t3-t2))
print("M4: %f"%(t4-t3))
print("M5: %f"%(t5-t4))
print("M6: %f"%(t6-t5))