import torch
import torch.optim as optim
import gym
import threading
import time
import os
import numpy as np

from TrainVisualizer import TrainTracker
from network_architectures import PolicyNet, ValueNet
from ReinforcementTrainer import ReinforcementTrainer as RLTrainer


if __name__ == "__main__":
    # Testing Net
    environmentName = 'CartPole-v0'  #'LunarLander-v2' 
    use_baseline = True   # hyper 3 parameter
    num_episodes = 500
    rollout_limit = 500
    discount_factor = 1.0
    val_freq = 100
    hiddenSize = 64
    reps = 10
    lrs = np.logspace(-10, -3, num=8, base=2)
    bss = np.array(np.logspace(1, 6, num=6, base=2), dtype=int)

    env = gym.make(environmentName)
    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n

    for i in range(len(lrs)):
        for j in range(len(bss)):
            for k in range(reps):
                learning_rate = lrs[i]
                batch_size = bss[j]

                print('Search status: %d%%'%(100*(i*len(bss)*reps + j*reps + k)/(len(lrs)*len(bss)*reps)))
                print('  learning_rate = %f'%learning_rate)
                print('  batch_size    = %d'%batch_size)

                # setup policy network
                policy = PolicyNet(nInputs, hiddenSize, nActions)

                optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

                # train policy network
                trainer = RLTrainer(policy, env, optimizer)
                trainer.rollout_limit = rollout_limit
                trainer.discount_factor = discount_factor

                thread = threading.Thread(target=trainer.train, kwargs={'num_episodes':num_episodes, 'batch_size':batch_size, 'use_baseline':use_baseline})
                thread.start()
                time.sleep(1)

                # track training
                output_folder = r"C:\Users\ToreH\OneDrive - Københavns Universitet\Skole\02456 Deep Learning\Project\DeepLearningProject\output"
                outfiles = [os.path.join(output_folder, 'trainR_%2d_%f_%2d.csv'%(batch_size, learning_rate, k)), 
                            os.path.join(output_folder, 'valR_%2d_%f_%2d.csv'%(batch_size, learning_rate, k))]

                tv = TrainTracker([trainer.dataTrain, trainer.dataEval], thread, smooth_alphas=[0.05, 1], out_filepaths=outfiles)
                tv.start(update_interval=0.25)
