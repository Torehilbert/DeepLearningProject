import gym
import torch
import threading
import os
import sys
import time
import argparse
from datetime import datetime

from network_policy import Policy
from network_critic import Critic
from actor_critic import ActorCriticTrainer as ActorCriticTrainer

shared_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_SHARED')
sys.path.append(shared_path)

from Tracker import Tracker


INITIAL_POLICY_NAME = "policy.pt"
INITIAL_CRITIC_NAME = "critic.pt"


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, type=str)
    parser.add_argument('--episodes', '--n', required=True, type=int)

    parser.add_argument('--ini_path', '--ipath', required=False, type=str, default=None)
    parser.add_argument('--out_path', '--opath', required=False, type=str, default=None)

    parser.add_argument('--entropy_weight', '--entw', required=False, type=float, default=0.01)
    parser.add_argument('--gamma', '--discount_factor', '--g', required=False, type=float, default=0.99)
    parser.add_argument('--rollout_limit', '--rol', default=2000, required=False, type=int)
    parser.add_argument('--hiddensize', '--hs', required=False, type=int, default=128)
    parser.add_argument('--lr_policy', '--lrp', required=False, type=float, default=1e-4)
    parser.add_argument('--lr_critic', '--lrc', required=False, type=float, default=1e-3)
    parser.add_argument('--updrate_target', '--updt', required=False, type=int, default=10)
    parser.add_argument('--vrate', '--vr', required=False, type=int, default=200)
    parser.add_argument('--vcount', '--vc', required=False, type=int, default=10)
    args = parser.parse_args()

    if(args.out_path is None):
        output_parent = r"C:\Source\DeepLearningProject\Outputs"
        folder_name = args.env + " AC " + datetime.now().strftime("(%Y-%m-%d) (%H-%M-%S)")
        output_path = os.path.join(output_parent, folder_name)
        args.out_path = output_path

    # Script
    env = gym.make(args.env)  # 'LunarLander-v2' 'CartPole-v0'
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = Policy(n_inputs, n_actions, args.hiddensize)
    critic = Critic(n_inputs, args.hiddensize)

    if(args.ini_path is not None):
        path_policy = os.path.join(args.ini_path, INITIAL_POLICY_NAME)
        policy.load_state_dict(torch.load(path_policy))
        path_critic = os.path.join(args.ini_path, INITIAL_CRITIC_NAME)
        critic.load_state_dict(torch.load(path_critic))

    print('start')
    kwargs = {
        'lr_policy': args.lr_policy,        # 1e-4
        'lr_critic': args.lr_critic,        # 1e-3
        'use_separate_target': True,
        'target_update_rate': args.updrate_target,
        'gamma': args.gamma,
        'entropy_weight': args.entropy_weight,
        'max_grad_norm_policy': None,
        'max_grad_norm_critic': None,
        'number_of_episodes': args.episodes,
        'rollout_limit': args.rollout_limit,
        'validation_rate': args.vrate,
        'validation_count': args.vcount,
        'validation_rpint': True,
        'path_output': args.out_path
    }
    trainer = ActorCriticTrainer(policy, critic, env, **kwargs)

    thread = threading.Thread(target=trainer.train)
    thread.start()

    time.sleep(1)

    f = open(os.path.join(args.out_path, 'info.txt'), 'w')
    f.write('environment = ' + args.env + "\n")
    f.write('hiddensize = ' + str(args.hiddensize) + "\n")
    f.write('initial model = ' + str(args.ini_path) + "\n")

    for key in kwargs:
        f.write(key + " = " + str(kwargs[key]) + "\n")

    f.close()
