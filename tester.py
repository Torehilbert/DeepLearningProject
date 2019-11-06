import torch

A = torch.ones([20, 2])
A[:, 1] = 2

indx = torch.randint(2, [20, 1])
print(A)
print(indx)
print(A.gather(1, indx))
