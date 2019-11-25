import gym
import torch
import torch.optim as optim
import copy
import numpy as np
import random
import math


def train(policy_global, critic_global, steps_global, episode_count, steps_lock, args):

    env = gym.make(args.env)

    policy_local = copy.deepcopy(policy_global)
    critic_local = copy.deepcopy(critic_global)

    optimizer_policy = optim.Adam(policy_global.parameters(), lr=args.lr_policy)
    optimizer_critic = optim.Adam(critic_global.parameters(), lr=args.lr_critic)

    policy_local.train()
    critic_local.train()

    state = env.reset()
    done = True
    first = True

    epslen = 0
    while (1):
        policy_local.load_state_dict(policy_global.state_dict())
        critic_local.load_state_dict(critic_global.state_dict())

        states = []
        policy_dists = []
        actions = []
        rewards = []

        for step in range(args.num_steps):
            epslen += 1
            state = torch.from_numpy(np.atleast_2d(state)).float()
            policy_dist = policy_local(state)
            action = (random.random() < np.cumsum(np.squeeze(policy_dist.detach().numpy()))).argmax()
            state_next, reward, done, _ = env.step(action)

            states.append(state)
            policy_dists.append(policy_dist)
            actions.append(action)
            rewards.append(reward)
            if(epslen >= args.rollout_limit):
                done = True
            if(done):
                with steps_lock:
                    episode_count.value += 1
                    steps_global.value += epslen
                state = env.reset()
                epslen = 0
                break

            state = state_next

        with steps_lock:
            steps_global.value += (step + 1)

        states = torch.stack(states).squeeze(1)
        policy_dists = torch.stack(policy_dists).squeeze(1)
        actions = torch.from_numpy(np.atleast_2d(actions)).long().view(-1, 1)

        values = critic_local(states)
        returns = np.zeros(shape=(len(rewards),))
        R = critic_local(torch.from_numpy(np.atleast_2d(state_next)).float()) if not done else 0  # boostrap
        for t in reversed(range(len(rewards))):
            R = rewards[t] + args.discount * R
            returns[t] = R
        returns = torch.from_numpy(returns).float().view(-1, 1)
        advantages = returns - values

        loss_critic = (advantages * advantages).sum()

        log_policy_dists = torch.log(policy_dists + 1e-9)
        log_probs = log_policy_dists.gather(1, actions)
        loss_policy = -(log_probs * advantages.detach()).sum()

        entropy = -(log_policy_dists * policy_dists).sum()
        loss_entropy = -args.entropy_weight * entropy

        loss = loss_policy + loss_entropy
        optimizer_policy.zero_grad()
        optimizer_critic.zero_grad()

        loss_critic.backward()
        loss.backward()

        if(first):
            for param_local, param_global in zip(policy_local.parameters(), policy_global.parameters()):
                param_global._grad = param_local.grad
            for param_local, param_global in zip(critic_local.parameters(), critic_global.parameters()):
                param_global._grad = param_local.grad
            first = False

        optimizer_policy.step()
        optimizer_critic.step()

        if(args.max_steps is not None and steps_global.value >= args.max_steps):
            break

        if(args.max_episodes is not None and episode_count.value >= args.max_episodes):
            break
