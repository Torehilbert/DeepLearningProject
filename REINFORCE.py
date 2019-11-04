import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Learn(policy, optimizer, rollouts, discount_factor, use_baseline=False):
    ''' REINFORCE learn algorithm

    params:
        policy              : torch module network
        optimizer           : torch optimizer
        rollouts            : list of lists of (s,a,r) tuples
        discount_factor     : ...
    
    returns:
        pass

    '''

    # Prepare data
    batch_size = len(rollouts)

    all_states = []
    all_actions = []
    all_rewards = []
    reward_sum = 0
    reward_sums = []
    all_returns = []
    lengths = []
    for i in range(batch_size):
        rollout = np.array(rollouts[i])
        lengths.append(len(rollout))

        states = np.vstack(rollout[:, 0])
        all_states.append(states)

        actions = np.vstack(rollout[:, 1])
        all_actions.append(actions)

        rewards = np.array(rollout[:, 2], dtype=float)
        all_rewards.append(rewards)
        reward_sum += np.sum(rewards)
        reward_sums.append(np.sum(rewards))

        returns = _expected_future_rewards(rewards, discount_factor)
        all_returns.append(returns)

    if(use_baseline):
        returns_baseline = np.zeros(shape=(max(lengths), batch_size))

        for i in range(len(all_returns)):
            returns_baseline[0:len(all_returns[i]), i] = all_returns[i]

        returns_baseline = np.mean(returns_baseline, axis=1)

        for i in range(len(all_returns)):
            all_returns[i] = all_returns[i] - returns_baseline[0:len(all_returns[i])]

    optimizer.zero_grad()
    loss = 0
    for i in range(batch_size):
        states = all_states[i]
        actions = all_actions[i]
        returns = all_returns[i]

        action_probabilities = policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
        loss += _loss(action_probabilities, returns)
    loss.backward()
    optimizer.step()

    return reward_sums, 0
    # rollout = np.array(rollout)
    # states = np.vstack(rollout[:, 0])
    # actions = np.vstack(rollout[:, 1])
    # rewards = np.array(rollout[:, 2], dtype=float)
    # returns = _expected_future_rewards(rewards, discount_factor)

    # optimizer.zero_grad()
    # action_probabilities = policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
    # loss = _loss(action_probabilities, returns)
    # loss.backward()
    # optimizer.step()

    #return np.sum(rewards), loss.item()


def _expected_future_rewards(rewards, discount_factor):
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor*returns[t+1]
    return returns


def _loss(action_probabilities, returns):
    return -torch.mean(torch.mul(torch.log(action_probabilities), torch.from_numpy(returns).float()))

