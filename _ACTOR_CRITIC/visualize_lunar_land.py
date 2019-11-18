import gym
import torch
import numpy as np
import time
import argparse
import os

from network_policy import Policy


DEFAULT_EXPLORE = False
DEFAULT_ROLLOUT_LIMIT = 1500
DEFAULT_REPS = 10


def extract_info_data(path):
    f = open(path, 'r')
    cont = f.read()
    f.close()
    info_data = {}
    for line in cont.split("\n"):
        splits = line.split(" = ")
        if(len(splits) > 1):
            info_data[splits[0]] = splits[1]
    return info_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--rollout_limit', required=False, type=int, default=DEFAULT_ROLLOUT_LIMIT)
    parser.add_argument('--explore', required=False, type=bool, default=DEFAULT_EXPLORE)
    parser.add_argument('--reps', required=False, type=int, default=DEFAULT_REPS)
    args = parser.parse_args()

    policy_path = os.path.join(args.data, 'policy.pt')
    info_data = extract_info_data(os.path.join(args.data, 'info.txt'))

    environment_name = info_data['environment']
    hiddensize = int(info_data['hiddensize'])

    # Render
    env = gym.make(environment_name)
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = Policy(n_inputs, n_actions, hiddensize)
    policy.load_state_dict(torch.load(policy_path))

    total_rewards = []
    for rep in range(args.reps):
        rewards = []
        state = env.reset()
        for i in range(args.rollout_limit):
            env.render()
            action = policy.get_action(state=state, explore=args.explore)
            state_next, reward, done, _ = env.step(action)
            state = state_next
            rewards.append(reward)
            if done:
                break
        total_rewards.append(np.sum(rewards))
        print(" %d Total reward: " % (rep + 1), total_rewards[-1])
    print("Mean total reward: ", np.mean(total_rewards))
    time.sleep(3)
    env.close()
