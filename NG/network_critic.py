import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, inFeatures):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(inFeatures, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)
