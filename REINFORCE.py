import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Learn(policy, optimizer, rollout, discount_factor):
    ''' REINFORCE learn algorithm

    params:
        policy      : torch module network
        optimizer   : torch optimizer
        rollout     : list of (s,a,r) tuples
    
    returns:
        pass

    '''

    rollout = np.array(rollout)
    states = np.vstack(rollout[:, 0])
    actions = np.vstack(rollout[:, 1])
    rewards = np.array(rollout[:, 2], dtype=float)
    returns = _expected_future_rewards(rewards, discount_factor)

    optimizer.zero_grad()
    action_probabilities = policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
    loss = _loss(action_probabilities, returns)
    loss.backward()
    optimizer.step()

    return np.sum(rewards), loss.item()


def _expected_future_rewards(rewards, discount_factor):
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor*returns[t+1]
    return returns


def _loss(action_probabilities, returns):
    return -torch.mean(torch.mul(torch.log(action_probabilities), torch.from_numpy(returns).float()))
