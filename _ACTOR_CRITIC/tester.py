
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import sys
import time
import gym 

from network_policy import Policy



if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    print(env.action_space.shape)
