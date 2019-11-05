import torch
import torch.optim as optim
import gym
import threading
import time
import os

from Tracker import Tracker
from network_architectures import PolicyNet, PolicyNetDouble
from Trainer import Trainer as trainer


LUNAR_LANDER = 'LunarLander-v2'
CART_POLE = 'CartPole-v0'

if __name__ == "__main__":
    # Parameters
    output_folder = r"C:\Source\DeepLearningProject\output"
    initial_model = (False, r"C:\Source\DeepLearningProject\Input\model_initial.pt")
    final_model = (True, r"C:\Source\DeepLearningProject\output\model_final.pt")
    os.makedirs(output_folder, exist_ok=True)

    hiddenSizes = [64, 64]
    number_of_batches = 2500
    batch_size = 10
    validation_frequency = 2
    learning_rate = 0.002
    use_baseline = True
    rollout_limit = 500
    discount_factor = 1.0

    # Environment
    environmentName = CART_POLE
    env = gym.make(environmentName)

    # Policy
    policy = PolicyNetDouble(env.observation_space.shape[0], hiddenSizes, env.action_space.n)
    if(initial_model[0]):
        policy.load_state_dict(torch.load(initial_model[1]))

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Trainer
    trainer = trainer(policy=policy,
                        environment=env,
                        optimizer=optimizer,
                        rollout_limit=rollout_limit,
                        discount_factor=discount_factor,
                        model_path_save=final_model[1] if final_model[0] else None)

    thread = threading.Thread(target=trainer.train,
                                kwargs={'number_of_batches': number_of_batches,
                                        'batch_size': batch_size,
                                        'val_freq': validation_frequency,
                                        'use_baseline': use_baseline})
    thread.start()
    time.sleep(1)

    # Tracker (save training data to files and show plots etc)
    csv_train_rewards = os.path.join(output_folder, 'train_reward.csv')
    csv_eval_rewards = os.path.join(output_folder, 'validation_reward.csv')
    tv = Tracker(data_containers=[trainer.data_buffer_train, trainer.data_buffer_eval],
                        train_thread=thread,
                        smooth_alphas=[0.05, 1],
                        out_filepaths=[csv_train_rewards, csv_eval_rewards])
    tv.start(update_interval=0.25)
