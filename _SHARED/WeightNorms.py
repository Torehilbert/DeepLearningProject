import torch
import sys
import os

asset_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
asset_path = os.path.join(asset_path, '_ACTOR_CRITIC')
sys.path.append(asset_path)

from network_policy import Policy
from network_critic import Critic



model_path = r"C:\Source\DeepLearningProject\Outputs\LunarLander-v2 AC (2019-11-11) (20-42-07) Looking good\policy.pt"

policy = Policy(8, 4, 64)
policy.load_state_dict(torch.load(model_path))

for param in policy.parameters():
    norm = torch.norm(param.data)
    print(norm)
