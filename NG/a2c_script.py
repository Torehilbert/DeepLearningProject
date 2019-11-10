import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import pandas as pd
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from network_policy import Policy
from network_critic import Critic
from a2c_algo import A2CTrainer
from Tracker import Tracker

LUNAR_LANDER = 'LunarLander-v2'
CART_POLE = 'CartPole-v0'


if __name__ == "__main__":
    env = gym.make(LUNAR_LANDER)
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = Policy(n_inputs, n_actions)
    critic = Critic(n_inputs)

    #policy.load_state_dict(torch.load(r"C:\Source\DeepLearningProject\output CP Pretrained\policy.pt"))
    #critic.load_state_dict(torch.load(r"C:\Source\DeepLearningProject\output CP Pretrained\critic.pt"))
    

    print('start')
    kwargs = {
        'lr_policy': 1e-3,  # 1e-3
        'lr_critic': 1e-2,  # 1e-2
        'gamma': 0.99,
        'entropy_weight': 0.001,
        'max_grad_norm_policy': 0.5,
        'max_grad_norm_critic': 0.5,
        'number_of_episodes': 10000,
        'rollout_limit': 500,
        'validation_rate': 1000,
        'validation_count': 5,
        'validation_rpint': True,
    }
    trainer = A2CTrainer(policy, critic, env, **kwargs)

    thread = threading.Thread(target=trainer.train)
    thread.start()
   
    # tracker
    csv_train_rewards = os.path.join(trainer.path_tracks_folder, 'train_reward.csv')
    csv_eval_rewards = os.path.join(trainer.path_tracks_folder, 'validation_reward.csv')
    csv_loss_policy = os.path.join(trainer.path_tracks_folder, 'loss_actor.csv')
    csv_loss_critic = os.path.join(trainer.path_tracks_folder, 'loss_critic.csv')
    csv_loss_entropy = os.path.join(trainer.path_tracks_folder, 'loss_entropy.csv')

    
    # tv = Tracker(
    #     data_containers=[
    #         trainer.data_buffer_train,
    #         trainer.data_buffer_eval,
    #         trainer.data_buffer_loss_policy,
    #         trainer.data_buffer_loss_critic,
    #         trainer.data_buffer_loss_entropy],
    #     train_thread=thread,
    #     smooth_alphas=[0.03, 1, 0.03, 0.03, 0.03],
    #     out_filepaths=[
    #         csv_train_rewards,
    #         csv_eval_rewards,
    #         csv_loss_policy,
    #         csv_loss_critic,
    #         csv_loss_entropy]
    # )
    # tv.initialize()
    # tv.format(id=0, ylabel='R (train)')
    # tv.format(id=1, ylabel='R (val)')
    # tv.format(id=2, ylabel='L (actor)')
    # tv.format(id=3, ylabel='L (critic)')
    # tv.format(id=4, ylabel='L (entropy)')
    # tv.start(update_interval=1)
