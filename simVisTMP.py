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
    environmentName = 'LunarLander-v2'  #'CartPole-v0'  #
    folder = r"C:\Users\ToreH\OneDrive - Københavns Universitet\Skole\02456 Deep Learning\Project\DeepLearningProject\output_llv2_models"
    model = "model_1920.pt"
    policy_file_path = os.path.join(folder,model)
    
    

    env = gym.make(environmentName)

    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n
    hiddenSize = 128
    policy = PolicyNet(nInputs, hiddenSize, nActions)
    policy.load_state_dict(torch.load(policy_file_path))

    s = env.reset()
    for _ in range(3000):
        env.render()

        a, _ = policy.get_action(state=s, explore=False)
        print(a)
        s1,r,done,_ = env.step(a)
        s=s1
        if done:
            break
    
    time.sleep(1)
    env.close()