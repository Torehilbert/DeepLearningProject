import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import gym
import threading
import time
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path)

from Tracker import Tracker
from Architectures import PolicyNetDouble, ActorCritic


def debug_norm_evolution(models_path=r"C:\Source\DeepLearningProject\output\models", environment_name='LunarLander-v2'):
    env = gym.make(environment_name)
    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n

    models = os.listdir(models_path)
    all_norms = []
    for i in range(len(models)):
        network = ActorCritic(nInputs, nActions)
        network.load_state_dict(torch.load(os.path.join(models_path, models[i])))

        norms = []
        for param in network.parameters():
            norms.append(param.data.norm(p=2))
        all_norms.append(norms)

    
    return np.array(all_norms)
        
if __name__ == "__main__":
    env_name = 'LunarLander-v2'
    path = r"C:\Source\DeepLearningProject\output DEBUG\models"
    norms = debug_norm_evolution(path, env_name)
    print(norms)
    plt.figure()
    plt.plot(norms)
    plt.legend(list(range(norms.shape[1])))
    plt.show()
