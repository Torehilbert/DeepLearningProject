import torch
import torch.optim as optim
import gym
import threading
import time
import os

from Tracker import Tracker
from Architectures import PolicyNetDouble, ActorCritic


LUNAR_LANDER = 'LunarLander-v2'
CART_POLE = 'CartPole-v0'

if __name__ == "__main__":
    environmentName = LUNAR_LANDER
    #folder = r"C:\Source\DeepLearningProject\output_lunar_lander_solved_250k"
    folder = r"C:\Source\DeepLearningProject\output"
    model = "model_final.pt"
    policy_file_path = os.path.join(folder, model)

    env = gym.make(environmentName)

    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n
    hiddenSizes = [64, 64]
    #policy = PolicyNetDouble(nInputs, hiddenSizes, nActions)
    policy = ActorCritic(nInputs, hiddenSizes, nActions)
    policy.load_state_dict(torch.load(policy_file_path))

    s = env.reset()
    for _ in range(3000):
        env.render()

        a, _ = policy.get_action(state=s, explore=False)
        s1, r, done, _ = env.step(a)
        s = s1
        if done:
            break
    
    time.sleep(1)
    env.close()