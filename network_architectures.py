import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, inFeatures, hiddenSize, outFeatures):
        super(PolicyNet, self).__init__()

        self.l1 = nn.Linear(inFeatures, hiddenSize, bias=False)
        self.l2 = nn.Linear(hiddenSize, outFeatures)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x), dim=1)

    def get_action(self, state, explore=False):
        with torch.no_grad():
            actionProbabilities = self(torch.from_numpy(np.atleast_2d(state)).float())
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


class ValueNet(nn.Module):
    def __init__(self, inFeatures, hiddenSizes):
        super(ValueNet, self).__init__()
        if(len(hiddenSizes) != 2):
            raise Exception("This ValueNet assumes 2 hidden layers!")

        self.l1 = nn.Linear(inFeatures, hiddenSizes[0])
        self.l2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.l3 = nn.Linear(hiddenSizes[1], 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)