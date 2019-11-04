import torch
import numpy as np
import os

from TimeManager import TimeManager as TM
from Data import Data
import REINFORCE


class Trainer:
    def __init__(self, policy, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        self.policy = policy
        self.env = environment
        self.optimizer = optimizer
        self.rollout_limit = rollout_limit
        self.discount_factor = discount_factor
        self.model_path_save = model_path_save

        self.data_buffer_train = Data()
        self.data_buffer_eval = Data()

    def train(self, number_of_batches=100, batch_size=10, val_freq=10, use_baseline=False):
        print("Training started...")
        for batch in range(number_of_batches):
            self.policy.train()
            rollouts = self._simulate_multiple_rollouts(batch_size, self.rollout_limit)

            batch_rewards, loss = REINFORCE.Learn(self.policy, self.optimizer, rollouts, self.discount_factor, use_baseline)
            self.data_buffer_train.add_data(batch_rewards)

            if (((batch + 1) % val_freq) == 0):
                validation_reward = self.validate()
                print(' %d%% : Validation reward: %d' % ((100 * (batch + 1) / number_of_batches), validation_reward))
        self._save_model()
        print("Training ended")

    def validate(self, count=10):
        self.policy.eval()
        with torch.no_grad():
            rollouts = self._simulate_multiple_rollouts(number_of_rollouts=count, rollout_limit=self.rollout_limit)

        reward = 0
        for i in range(len(rollouts)):
            reward += np.sum(np.array(rollouts[i])[:, 2])
        reward = reward / len(rollouts)
        self.data_buffer_eval.add_data([reward])
        self._save_model()
        return reward

    def simulate_single_rollout(self, rollout_limit):
        rollout = []
        s = self.env.reset()
        for i in range(rollout_limit):
            a, _ = self.policy.get_action(state=s, explore=True)
            s1, r, done, _ = self.env.step(a)
            rollout.append((s, a, r))
            s = s1
            if done:
                break
        return rollout

    def _simulate_multiple_rollouts(self, number_of_rollouts, rollout_limit):
        return [self.simulate_single_rollout(rollout_limit) for i in range(number_of_rollouts)]

    def _save_model(self):
        if(self.model_path_save is not None):
            torch.save(self.policy.state_dict(), self.model_path_save)
