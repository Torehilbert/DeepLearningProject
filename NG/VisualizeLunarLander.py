import gym
import torch
import numpy as np
import time
import sys

from network_policy import Policy
from network_critic import Critic


DEFAULT_ENVIRONMENT = 'LunarLander-v2'
DEFAULT_POLICY_PATH = r"C:\Source\DeepLearningProject\output\policy.pt"
DEFAULT_EXPLORE = False
DEFAULT_ROLLOUT_LIMIT = 1500
DEFAULT_REPS = 1

if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    environment_name = args[0] if len(args) >= 1 else DEFAULT_ENVIRONMENT
    policy_path = args[1] if len(args) >= 2 else DEFAULT_POLICY_PATH
    explore = bool(int(args[2])) if len(args) >= 3 else DEFAULT_EXPLORE
    rollout_limit = int(args[3]) if len(args) >= 4 else DEFAULT_ROLLOUT_LIMIT
    reps = int(args[4]) if len(args) >= 5 else DEFAULT_REPS

    # Render
    env = gym.make(environment_name)
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = Policy(n_inputs, n_actions)
    policy.load_state_dict(torch.load(policy_path))

    total_rewards = []
    for rep in range(reps):
        rewards = []
        state = env.reset()
        for i in range(rollout_limit):
            env.render()
            action = policy.get_action(state=state, explore=explore)
            state_next, reward, done, _ = env.step(action)
            state = state_next
            rewards.append(reward)
            if done:
                break
        total_rewards.append(np.sum(rewards))
        print(" %d Total reward: "% (rep+1), total_rewards[-1])
    print("Mean total reward: ", np.mean(total_rewards))
    time.sleep(3)
    env.close()
