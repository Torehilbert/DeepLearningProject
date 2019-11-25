
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import sys
import time

from network_policy import Policy


# def ensure_shared_grads(model, shared_model):
#     for param, shared_param in zip(model.parameters(), shared_model.parameters()):
#         if shared_param.grad is not None:
#             return
#         shared_param._grad = param.grad

def value_from_one_process_view(model):
    counter = 0
    while(counter < 5):
        counter += 1
        for p in model.parameters():
            print("data: ", p.data)
            break
        time.sleep(1)


if __name__ == "__main__":
    policy = Policy(2, 2, 3)
    policy.share_memory()

    p = mp.Process(target=value_from_one_process_view, args=(policy,))
    p.start()

    A = torch.zeros([3, 2])
    time.sleep(0.5)
    for param in policy.parameters():
        param.data[0, 0] = 100
        break

    p.join()
    print('done')
