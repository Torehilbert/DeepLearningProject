import torch
import torch.optim as optim
import gym
import threading
import time
import os

from TrainVisualizer import TrainTracker
from network_architectures import PolicyNet, ValueNet
from ReinforcementTrainer import ReinforcementTrainer as RLTrainer


if __name__ == "__main__":
    output_folder = r"C:\Users\ToreH\OneDrive - Københavns Universitet\Skole\02456 Deep Learning\Project\DeepLearningProject\output"

    # Testing Net
    environmentName = 'LunarLander-v2'  #'CartPole-v0'  #
    hiddenSize = 128

    env = gym.make(environmentName)

    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n

    # training settings
    num_episodes = 2000
    batch_size = 8         # hyper 1 parameter #10 is good
    lr_policy = 0.01        # hyper 2 parameter #0.05 is good ()
    use_baseline = True     # hyper 3 parameter

    rollout_limit = 500  # max rollout length
    discount_factor = 0.98  # reward discount factor (gamma), 1.0 = no discount

    val_freq = 100  # validation frequency

    # setup policy network
    policy = PolicyNet(nInputs, hiddenSize, nActions)

    optimizer = optim.Adam(policy.parameters(), lr=lr_policy)

    # train policy network
    trainer = RLTrainer(policy, env, optimizer)
    trainer.rollout_limit = rollout_limit
    trainer.discount_factor = discount_factor

    thread = threading.Thread(target=trainer.train, kwargs={'num_episodes':num_episodes, 'batch_size':batch_size, 'use_baseline':use_baseline, 'model_output_folder':output_folder})
    thread.start()
    time.sleep(1)

    # track training
    outfiles = [os.path.join(output_folder, 'train_reward.csv'), 
                os.path.join(output_folder, 'validation_reward.csv')]

    tv = TrainTracker([trainer.dataTrain, trainer.dataEval], thread, smooth_alphas=[0.05, 1], out_filepaths=outfiles)
    tv.initialize()
    tv.format(0, 'Iteration (training)', 'Training reward', [0, num_episodes])
    tv.format(1, 'Iteration (validation)', 'Validation reward', [0, num_episodes//val_freq])
    tv.start(update_interval=0.25)
