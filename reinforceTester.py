import sys
print(sys.executable)

import torch
import torch.optim as optim
import numpy as np
import gym
import threading
import matplotlib.pyplot as plt
import time

import REINFORCE as R

from TrainVisualizer import TrainTracker

from network_architectures import PolicyNet, ValueNet
from ReinforcementTrainer import ReinforcementTrainer as RLTrainer

if __name__ == "__main__":
    # Testing Net
    environmentName = 'CartPole-v0'  #'LunarLander-v2' 
    hiddenSize = 512

    env = gym.make(environmentName)

    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n

    # training settings
    num_episodes = 1000
    rollout_limit = 500  # max rollout length
    discount_factor = 1.0  # reward discount factor (gamma), 1.0 = no discount
    learning_rate = 0.001  # you know this by now
    val_freq = 100  # validation frequency

    # setup policy network
    valueNet = ValueNet(nInputs, 512)
    policy = PolicyNet(nInputs, hiddenSize, nActions)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # train policy network
    trainer = RLTrainer(policy, env, optimizer)
    trainer.rollout_limit = rollout_limit
    trainer.discount_factor = discount_factor

    thread = threading.Thread(target=trainer.train, kwargs={'num_episodes':num_episodes})
    thread.start()
    time.sleep(1)

    # follow training
    tv = TrainTracker([trainer.dataTrain, trainer.dataEval], thread, smooth_alphas=[0.02, 1])
    tv.initialize()
    tv.format(0, 'Iteration (training)', 'Training reward', [0, num_episodes])
    tv.format(1, 'Iteration (validation)', 'Validation reward', [0, num_episodes//val_freq - 1])
    tv.start(update_interval=0.25)

    print('done')
