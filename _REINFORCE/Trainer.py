import torch
import numpy as np
import os

from Data import Data
import REINFORCE


class BaseTrainer:
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, path_output=None):
        self.network = network
        self.env = environment
        self.optimizer = optimizer
        self.rollout_limit = rollout_limit
        self.discount_factor = discount_factor

        self.data_buffer_train = Data()
        self.data_buffer_eval = Data()

        self.path_output_folder = path_output
        self.path_models_folder = os.path.join(path_output, 'models') if path_output is not None else None
        self.path_tracks_folder = os.path.join(path_output, 'tracks') if path_output is not None else None
        self.path_final_model = os.path.join(path_output, 'model.pt') if path_output is not None else None
        self.path_info_file = os.path.join(path_output, 'info.txt') if path_output is not None else None

        os.makedirs(self.path_output_folder, exist_ok=True)
        os.makedirs(self.path_models_folder, exist_ok=True)
        os.makedirs(self.path_tracks_folder, exist_ok=True)

    def train():
        raise NotImplementedError()

    def simulate_rollout(self, rollout_limit, exploration=False):
        rollout = []
        s = self.env.reset()
        for i in range(rollout_limit):
            a, _ = self.network.get_action(state=s, explore=exploration)
            s1, r, done, _ = self.env.step(a)
            rollout.append((s, a, r))
            s = s1
            if done:
                break
        return rollout 

    def validate(self, count=10, index=None):
        self.network.eval()
        with torch.no_grad():
            rollouts = [self.simulate_rollout(self.rollout_limit, False) for i in range(count)]

        total_rewards = []
        for i in range(len(rollouts)):
            total_rewards.append(np.sum(np.array(rollouts[i])[:, 2]))
        self.data_buffer_eval.add_data(total_rewards)

        self._save_model(path=os.path.join(self.path_models_folder, '%d.pt' % index))
        self._save_model()
        self._save_info()
        return np.mean(total_rewards)

    def _save_model(self, path=None, state_dict=None):
        if(state_dict is None):
            state_dict = self.network.state_dict()

        if(path is not None):
            torch.save(state_dict, path)
        elif(self.path_final_model is not None):
            torch.save(state_dict, self.path_final_model)

    def _save_info(self):
        if(self.path_info_file is not None):
            env_name = self.env.unwrapped.spec.id
            f = open(self.path_info_file, 'w')
            f.write(env_name)
            f.close()


class REINFORCETrainer(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        super(REINFORCETrainer, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, model_path_save)

    def train(self, number_of_batches=100, batch_size=10, val_freq=10, use_baseline=False):
        print("Training started...")
        for batch in range(number_of_batches):
            self.network.train()
            rollouts = [self.simulate_rollout(self.rollout_limit, True) for i in range(batch_size)]

            batch_rewards, loss = REINFORCE.Learn(self.network, self.optimizer, rollouts, self.discount_factor, use_baseline)
            self.data_buffer_train.add_data(batch_rewards)

            if (((batch + 1) % val_freq) == 0):
                validation_reward = self.validate()
                print(' %d%% : Validation reward: %d' % ((100 * (batch + 1) / number_of_batches), validation_reward))
        self._save_model()
        print("Training ended")