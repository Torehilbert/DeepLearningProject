import torch
import torch.optim as optim
import gym
import threading
import time
import os

from Tracker import Tracker
from Architectures import ActorCriticStable as ActorCritic
from Trainer import A2CTrainerStable as trainer


LUNAR_LANDER = 'LunarLander-v2'
CART_POLE = 'CartPole-v0'



GAMMA = None

MAX_GRAD_NORM = None
LOSS_COEFFICIENT_ACTOR = None
LOSS_COEFFICIENT_CRITIC = None
LOSS_COEFFICIENT_ENTROPY = None


if __name__ == "__main__":
    # Parameters
    output_folder = r"C:\Source\DeepLearningProject\output"
    initial_model = (False, r"C:\Source\DeepLearningProject\Input\model_initial.pt")
    os.makedirs(output_folder, exist_ok=True)

    hiddenSizes = [64, 64]
    number_of_episodes = 10000
    validation_frequency = number_of_episodes // 100
    learning_rate = 1e-3
    rollout_limit = 500
    discount_factor = 0.95

    # Environment
    environmentName = CART_POLE
    env = gym.make(environmentName)

    # Policy
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if(initial_model[0]):
        policy.load_state_dict(torch.load(initial_model[1]))

    # Optimizer
    #optimizer = optim.SGD(policy.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Trainer
    trainer = trainer(network=policy,
                        environment=env,
                        optimizer=optimizer,
                        rollout_limit=rollout_limit,
                        discount_factor=discount_factor,
                        path_output=output_folder)

    thread = threading.Thread(target=trainer.train,
                                kwargs={'episodes': number_of_episodes,
                                        'val_freq': validation_frequency})
    thread.start()
    time.sleep(1)

    # Tracker (save training data to files and show plots etc)
    csv_train_rewards = os.path.join(trainer.path_tracks_folder, 'train_reward.csv')
    csv_eval_rewards = os.path.join(trainer.path_tracks_folder, 'validation_reward.csv')
    csv_loss_total = os.path.join(trainer.path_tracks_folder, 'loss_total.csv')
    csv_loss_actor = os.path.join(trainer.path_tracks_folder, 'loss_actor.csv')
    csv_loss_critic = os.path.join(trainer.path_tracks_folder, 'loss_critic.csv')
    csv_loss_entropy = os.path.join(trainer.path_tracks_folder, 'loss_entropy.csv')
    csv_gradnorm = os.path.join(trainer.path_tracks_folder, 'grad_norm.csv')

    tv = Tracker(data_containers=[trainer.data_buffer_train, trainer.data_buffer_eval, trainer.data_buffer_loss, trainer.data_buffer_actor_loss, trainer.data_buffer_critic_loss, trainer.data_buffer_entropy_loss, trainer.data_buffer_gradient_norm],
                        train_thread=thread,
                        smooth_alphas=[0.05, 1, 1, 1, 1, 1, 1],
                        out_filepaths=[csv_train_rewards, csv_eval_rewards, csv_loss_total, csv_loss_actor, csv_loss_critic, csv_loss_entropy, csv_gradnorm])
    tv.initialize()
    tv.format(id=0, ylabel='R (train)')
    tv.format(id=1, ylabel='R (val)')

    tv.format(id=2, ylabel='L (total)')
    tv.format(id=3, ylabel='L (actor)')
    tv.format(id=4, ylabel='L (critic)')
    tv.format(id=5, ylabel='L (entropy)')
    tv.format(id=6, ylabel='Norm (grad)')

    tv.start(update_interval=0.25)
