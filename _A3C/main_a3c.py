import gym 
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import argparse
import sys
import time

from train_a3c import train
from validate_a3c import validate, render

path_actor_critic = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_ACTOR_CRITIC')
sys.path.append(path_actor_critic)

from network_policy import Policy
from network_critic import Critic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=False, type=str, default='ATC-v0')
    parser.add_argument('--lr_policy', required=False, type=float, default=1e-4)
    parser.add_argument('--lr_critic', required=False, type=float, default=1e-3)
    parser.add_argument('--discount', required=False, type=float, default=0.99)
    parser.add_argument('--hiddensize', required=False, type=int, default=64)
    parser.add_argument('--num_steps', required=False, type=int, default=10)
    parser.add_argument('--num_envs', required=False, type=int, default=12)
    parser.add_argument('--train_reward_alpha', required=False, type=float, default=0.99)
    parser.add_argument('--rollout_limit', required=False, type=int, default=500)
    parser.add_argument('--entropy_weight', required=False, type=float, default=0.01)
    parser.add_argument('--entropy_weight_end', required=False, type=float, default=None)
    parser.add_argument('--validation_count', required=False, type=int, default=10)
    parser.add_argument('--render', required=False, type=bool, default=False)
    parser.add_argument('--render_pause', required=False, type=float, default=1)
    parser.add_argument('--output_path', required=False, type=str, default=None)
    parser.add_argument('--input_path', required=False, type=str, default=None)
    parser.add_argument('--max_steps', required=False, type=int, default=None)
    parser.add_argument('--max_episodes', required=False, type=int, default=5000)
    args = parser.parse_args()

    env = gym.make(args.env)

    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_global = Policy(n_inputs, n_actions, args.hiddensize)
    policy_global.share_memory()

    critic_global = Critic(n_inputs, args.hiddensize)
    critic_global.share_memory()

    if(args.input_path is not None):
        policy_path = os.path.join(args.input_path, 'policy.pt')
        critic_path = os.path.join(args.input_path, 'critic.pt')
        
        policy_global.load_state_dict(torch.load(policy_path))
        critic_global.load_state_dict(torch.load(critic_path))

    episode_count = mp.Value('i', 0)

    steps_global = mp.Value('i', 0)
    atr = mp.Value('d', 0)
    steps_lock = mp.Lock()

    processes = []

    p = mp.Process(target=validate, args=(policy_global, critic_global, steps_global, episode_count, args))
    p.start()
    processes.append(p)

    if(args.render):
        p = mp.Process(target=render, args=(policy_global, args))
        p.start()

    for i in range(args.num_envs):
        p = mp.Process(target=train, args=(policy_global, critic_global, steps_global, episode_count, steps_lock, args))
        p.start()
        processes.append(p)

    for i in range(len(processes)):
        p.join()
    
    time.sleep(2)
    print('Done')
