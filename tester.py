import numpy as np

lrs = np.logspace(-10, -3, num=8, base=2)
print(lrs)

bss = np.logspace(1, 6, num=6, base=2)
print(bss)
# 2^-10 ... 2^-3
