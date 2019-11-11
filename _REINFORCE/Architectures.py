import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, outFeatures)
        )

        self.critic = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        for w in self.actor:
            if isinstance(w, nn.Linear):
                nn.init.constant_(w.bias, 0.0)

        for w in self.critic:
            if isinstance(w, nn.Linear):
                nn.init.constant_(w.bias, 0.0)

    def forward(self, x):
        actor_logits = self.actor(x)
        value = self.critic(x)
        return F.softmax(actor_logits, dim=1), value

    def get_action(self, state, explore=False):
        with torch.no_grad():
            actionProbabilities, _ = self(torch.from_numpy(np.atleast_2d(state)).float())
        if(explore):
            action = (np.cumsum(actionProbabilities.numpy()) > np.random.rand()).argmax()
        else:
            action = (actionProbabilities.numpy()).argmax()

        return action, actionProbabilities


class ActorCriticStable(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(ActorCriticStable, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, outFeatures)
        )

        self.critic = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        for w in self.actor:
            if isinstance(w, nn.Linear):
                nn.init.constant_(w.bias, 0.0)

        for w in self.critic:
            if isinstance(w, nn.Linear):
                nn.init.constant_(w.bias, 0.0)

    def forward(self, x):
        actor_logits = self.actor(x)
        value = self.critic(x)
        return F.softmax(actor_logits, dim=1), value

    def get_action(self, state, explore=False):
        with torch.no_grad():
            actionProbabilities, _ = self(torch.from_numpy(np.atleast_2d(state)).float())
        if(explore):
            action = (np.cumsum(actionProbabilities.numpy()) > np.random.rand()).argmax()
        else:
            action = (actionProbabilities.numpy()).argmax()

        return action, actionProbabilities

class PolicyNetDouble(nn.Module):
    def __init__(self, inFeatures, hiddenSizes, outFeatures):
        super(PolicyNetDouble, self).__init__()
        self.l1 = nn.Linear(inFeatures, hiddenSizes[0])
        self.l2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.l3 = nn.Linear(hiddenSizes[1], outFeatures)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim=1)

    def get_action(self, state, explore=False):
        with torch.no_grad():
            actionProbabilities = self(torch.from_numpy(np.atleast_2d(state)).float())
        if(explore):
            action = (np.cumsum(actionProbabilities.numpy()) > np.random.rand()).argmax()
        else:
            action = (actionProbabilities.numpy()).argmax()

        return action, actionProbabilities