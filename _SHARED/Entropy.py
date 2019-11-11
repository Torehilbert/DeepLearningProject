import numpy as np


def Entropy(probs):
    return -np.sum(np.log(probs) * probs)


def GetMaxValue(n_actions):
    probs = np.zeros(shape=(n_actions,)) + (1e-9)
    probs[0] = 1.0 - n_actions * (1e-9)
    return Entropy(probs)


def GetMinValue(n_actions):
    probs = np.ones(shape=(n_actions,)) / n_actions
    return Entropy(probs)


if __name__ == "__main__":
    action_count = 4
    minv = GetMinValue(action_count)
    maxv = GetMaxValue(action_count)
    print(0.001*minv, maxv)