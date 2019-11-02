import torch
import torch.optim as optim
import numpy as np
import gym
import threading
import matplotlib.pyplot as plt
import time

import REINFORCE as R
from ReinforcementTrainer import ReinforcementTrainer as RLTrainer

if __name__ == "__main__":
    # Testing Net
    environmentName = 'LunarLander-v2' # 'CartPole-v0' 
    hiddenSize = 64

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
    policy = R.PolicyNet(nInputs, hiddenSize, nActions)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # train policy network
    trainer = RLTrainer(policy, env, optimizer)

    thread = threading.Thread(target=trainer.train)
    thread.start()

    plt.figure()
    while(thread.is_alive()):
        plt.clf()

        plt.subplot(2, 1, 1)
        plt.plot(trainer.data.data[0])
        plt.xlim([0, trainer.data.sizes[0]])
        plt.ylim([0, 210])

        plt.subplot(2, 1, 2)
        plt.plot([None] + trainer.data.data[2])
        plt.xlim([0, trainer.data.sizes[2]])
        plt.ylim([0, 210])

        plt.pause(0.25)

    print('done')

    plt.show()
    #trainer.train()
