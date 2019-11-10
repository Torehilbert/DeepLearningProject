import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, outFeatures)
        )

    def forward(self, x):
        actor_logits = self.layers(x)
        return F.softmax(actor_logits, dim=1)

    def get_action(self, state, explore=False):
        with torch.no_grad():
            policy_distribution = self(torch.from_numpy(np.atleast_2d(state)).float()).numpy()

        if(explore):
            action = (np.cumsum(policy_distribution) > np.random.rand()).argmax()
        else:
            action = (policy_distribution).argmax()

        return action
