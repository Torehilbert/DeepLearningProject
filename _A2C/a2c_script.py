import gym
import torch
import threading
import os
import sys
from datetime import datetime

from network_policy import Policy
from network_critic import Critic
from a2c_algo import A2CTrainer

shared_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_SHARED')
sys.path.append(shared_path)

from Tracker import Tracker


LUNAR_LANDER = 'LunarLander-v2'
CART_POLE = 'CartPole-v0'
ENVIRONMENT_NAME = LUNAR_LANDER
HIDDENSIZE = 128

LOAD_INITIAL_MODELS = False
INITIAL_MODELS_PATH = r"C:\Source\DeepLearningProject\Outputs\output LL Almost solved"
INITIAL_POLICY_NAME = "policy.pt"
INITIAL_CRITIC_NAME = "critic.pt"

PATH_OUTPUT_PARENT = r"C:\Source\DeepLearningProject\Outputs"
path_output_folder_name = ENVIRONMENT_NAME + " A2C " + datetime.now().strftime("(%Y-%m-%d) (%H-%M-%S)")
path_output = os.path.join(PATH_OUTPUT_PARENT, path_output_folder_name)


if __name__ == "__main__":
    env = gym.make(ENVIRONMENT_NAME)
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = Policy(n_inputs, n_actions, HIDDENSIZE)
    critic = Critic(n_inputs, HIDDENSIZE)

    if(LOAD_INITIAL_MODELS):
        path_policy = os.path.join(INITIAL_MODELS_PATH, INITIAL_POLICY_NAME)
        policy.load_state_dict(torch.load(path_policy))
        path_critic = os.path.join(INITIAL_MODELS_PATH, INITIAL_CRITIC_NAME)
        critic.load_state_dict(torch.load(path_critic))

    print('start')
    kwargs = {
        'lr_policy': 1e-3,  # 1e-3
        'lr_critic': 1e-3,  # 1e-2
        'gamma': 0.99,
        'entropy_weight': 0.001,
        'max_grad_norm_policy': 0.5,
        'max_grad_norm_critic': 0.5,
        'number_of_episodes': 5000,
        'rollout_limit': 2000,
        'validation_rate': 200,
        'validation_count': 5,
        'validation_rpint': True,
        'path_output': path_output
    }
    trainer = A2CTrainer(policy, critic, env, **kwargs)

    thread = threading.Thread(target=trainer.train)
    thread.start()

    # tracker
    csv_train_rewards = os.path.join(trainer.path_tracks_folder, 'train_reward.csv')
    csv_eval_rewards = os.path.join(trainer.path_tracks_folder, 'validation_reward.csv')
    csv_loss_policy = os.path.join(trainer.path_tracks_folder, 'loss_actor.csv')
    csv_loss_critic = os.path.join(trainer.path_tracks_folder, 'loss_critic.csv')
    csv_loss_entropy = os.path.join(trainer.path_tracks_folder, 'loss_entropy.csv')

    tv = Tracker(
        data_containers=[
            trainer.data_buffer_train,
            trainer.data_buffer_eval,
            trainer.data_buffer_loss_policy,
            trainer.data_buffer_loss_critic,
            trainer.data_buffer_loss_entropy],
        train_thread=thread,
        smooth_alphas=[0.03, 1, 0.03, 0.03, 0.03],
        out_filepaths=[
            csv_train_rewards,
            csv_eval_rewards,
            csv_loss_policy,
            csv_loss_critic,
            csv_loss_entropy]
    )
    tv.initialize()
    tv.format(id=0, ylabel='R (train)')
    tv.format(id=1, ylabel='R (val)')
    tv.format(id=2, ylabel='L (actor)')
    tv.format(id=3, ylabel='L (critic)')
    tv.format(id=4, ylabel='L (entropy)')
    tv.start(update_interval=1)
