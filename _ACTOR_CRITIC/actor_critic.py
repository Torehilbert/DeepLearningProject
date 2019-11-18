import numpy as np
import random
import torch
import torch.optim as optim
import os
import time
import sys
import copy

shared_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_SHARED')
sys.path.append(shared_path)

from Data import DataMaster


DEFAULT_LEARNING_RATE_POLICY = 1e-3
DEFAULT_LEARNING_RATE_CRITIC = 1e-4
DEFAULT_ENTROPY_WEIGHT = 0.001
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_MAX_GRAD_POLICY = None  # 0.5
DEFAULT_MAX_GRAD_CRITIC = None  # 0.5
DEFAULT_USE_SEPARATE_TARGET = True
DEFAULT_TARGET_UPDATE_RATE = 10

DEFAULT_NUMBER_OF_EPISODES = 500
DEFAULT_ROLLOUT_LIMIT = 500
DEFAULT_VALIDATION_RATE = 50
DEFAULT_VALIDATION_COUNT = 10
DEFAULT_VALIDATION_PRINT = True
DEFAULT_VALIDATION_SAVE_MODEL = False
DEFAULT_OUTPUT_PATH = r"C:\Source\DeepLearningProject\Outputs\latest"


# should sample from batch multiple environments! should we use updated value for advantage in policy update?


class BaseTrainer:
    def __init__(self, policy_network, environment, **kwargs):
        self.env = environment

        self.policy = policy_network
        self.lr_policy = kwargs['lr_policy'] if 'lr_policy' in kwargs else DEFAULT_LEARNING_RATE_POLICY
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        self.max_grad_norm_policy = kwargs['max_grad_norm_policy'] if 'max_grad_norm_policy' in kwargs else DEFAULT_MAX_GRAD_POLICY
        self.entropy_weight = kwargs['entropy_weight'] if 'entropy_weight' in kwargs else DEFAULT_ENTROPY_WEIGHT

        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else DEFAULT_DISCOUNT_FACTOR
        self.number_of_episodes = kwargs['number_of_episodes'] if 'number_of_episodes' in kwargs else DEFAULT_NUMBER_OF_EPISODES
        self.rollout_limit = kwargs['rollout_limit'] if 'rollout_limit' in kwargs else DEFAULT_ROLLOUT_LIMIT
        self.validation_rate = kwargs['validation_rate'] if 'validation_rate' in kwargs else DEFAULT_VALIDATION_RATE
        self.validation_count = kwargs['validation_count'] if 'validation_count' in kwargs else DEFAULT_VALIDATION_COUNT
        self.validation_print = kwargs['validation_print'] if 'validation_print' in kwargs else DEFAULT_VALIDATION_PRINT
        self._save_model_for_each_validation = kwargs['validation_save_model'] if 'validation_save_model' in kwargs else DEFAULT_VALIDATION_SAVE_MODEL
        self.path_output_folder = kwargs['path_output'] if 'path_output' in kwargs else DEFAULT_OUTPUT_PATH
        self.path_models_folder = os.path.join(self.path_output_folder, 'models')
        self.path_tracks_folder = os.path.join(self.path_output_folder, 'tracks')
        self.path_final_model_policy = os.path.join(self.path_output_folder, 'policy.pt')

        os.makedirs(self.path_output_folder, exist_ok=True)
        os.makedirs(self.path_models_folder, exist_ok=True)
        os.makedirs(self.path_tracks_folder, exist_ok=True)

        self._create_data_instance()

    def _create_data_instance(self):
        csv_paths = [
            os.path.join(self.path_tracks_folder, 'train_reward.csv'),
            os.path.join(self.path_tracks_folder, 'validation_reward.csv'),
            os.path.join(self.path_tracks_folder, 'loss_critic.csv'),
            os.path.join(self.path_tracks_folder, 'loss_policy.csv'),
            os.path.join(self.path_tracks_folder, 'loss_entropy.csv')]

        self.data = DataMaster(n=5, filepaths=csv_paths, update_rate=0.1, udp=True)

    def train(self):
        raise NotImplementedError()

    def validate(self, episode=None, episodes_total=None):
        with torch.no_grad():
            results = []
            for rep in range(self.validation_count):
                state = self.env.reset()
                rewards = []
                for i in range(self.rollout_limit):
                    action = self.policy.get_action(state, explore=False)
                    state, reward, done, _ = self.env.step(action)
                    rewards.append(reward)
                    if done:
                        break
                results.append(np.sum(rewards))

            validation_reward = np.mean(results)
            self.data.add_data(validation_reward, 1)
            # self.data_buffer_eval.add_data([validation_reward])
            if(self.validation_print):
                if(episode is not None and episodes_total is None):
                    print("%d.  Validation reward: %.3f" % (episode + 1, validation_reward))
                elif(episode is not None):
                    print("%d%%. Validation reward: %.3f" % (100 * (episode + 1) / episodes_total, validation_reward))
                else:
                    print('Validation reward: %.3f' % validation_reward)

            if(self._save_model_for_each_validation):
                self._save_model(network=self.policy, path=os.path.join(self.path_models_folder, 'P%d.pt' % (episode + 1)))
            self._save_model(network=self.policy, path=self.path_final_model_policy)
        return validation_reward

    def _save_model(self, network, path=None):
        state_dict = network.state_dict()
        torch.save(state_dict, path)


class ActorCriticTrainer(BaseTrainer):
    def __init__(self, policy_network, critic_network, environment, **kwargs):
        super(ActorCriticTrainer, self).__init__(policy_network, environment, **kwargs)

        self.critic = critic_network

        self.use_separate_target = kwargs['use_separate_target'] if 'use_separate_target' in kwargs else DEFAULT_USE_SEPARATE_TARGET
        self.target_update_rate = kwargs['target_update_rate'] if 'target_update_rate' in kwargs else DEFAULT_TARGET_UPDATE_RATE

        self.lr_critic = kwargs['lr_critic'] if 'lr_critic' in kwargs else DEFAULT_LEARNING_RATE_CRITIC
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.max_grad_norm_critic = kwargs['max_grad_norm_critic'] if 'max_grad_norm_critic' in kwargs else DEFAULT_MAX_GRAD_CRITIC

        self.path_final_model_critic = os.path.join(self.path_output_folder, 'critic.pt')

    def train(self, number_of_episodes=None, rollout_limit=None):
        t0 = time.time()
        number_of_episodes = self.number_of_episodes if number_of_episodes is None else number_of_episodes
        rollout_limit = self.rollout_limit if rollout_limit is None else rollout_limit

        # Set correct target network
        if(self.use_separate_target):
            target_network = copy.deepcopy(self.critic)
            counter_reset_target = 0
        else:
            target_network = self.critic

        # START
        for episode in range(number_of_episodes):
            rewards = []
            losses_critic = []
            losses_policy = []
            losses_entropy = []

            state = self.env.reset()
            state = torch.from_numpy(np.atleast_2d(state))
            for t in range(rollout_limit):
                # 1. Step and sample
                state = torch.from_numpy(np.atleast_2d(state)).float()
                policy_distribution = self.policy(state)
                action = (random.random() < np.cumsum(np.squeeze(policy_distribution.detach().numpy()))).argmax()
                state_next, reward, done, _ = self.env.step(action)
                state_next = torch.from_numpy(np.atleast_2d(state_next)).float()

                # 2. Advantage
                value = self.critic(state)
                value_next = target_network(state_next).detach() if not done else 0
                advantage = reward + self.gamma * value_next - value

                # 3. Critic optimization
                loss_critic = advantage * advantage
                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

                if(self.use_separate_target):
                    counter_reset_target += 1
                    if(counter_reset_target == self.target_update_rate):
                        counter_reset_target = 0
                        target_network.load_state_dict(self.critic.state_dict())

                # 4. Policy optimization
                log_policy_distribution = torch.log(policy_distribution + 1e-9)
                entropy = -(log_policy_distribution * policy_distribution).sum()
                loss_entropy = -self.entropy_weight * entropy
                log_prob = log_policy_distribution[0, action]
                loss_policy = -log_prob * advantage.detach()
                loss = loss_policy + loss_entropy
                self.optimizer_policy.zero_grad()
                loss.backward()
                self.optimizer_policy.step()

                # 5. Bookkeeping
                state = state_next
                rewards.append(reward)
                losses_critic.append(loss_critic.detach().numpy())
                losses_policy.append(loss_policy.detach().numpy())
                losses_entropy.append(loss_entropy.detach().numpy())
                if(done):
                    break

            # validation
            if((episode + 1) % self.validation_rate == 0):
                self.validate_custom(episode, number_of_episodes)

            self.data.add_data_array(data=[np.sum(rewards), np.mean(losses_critic), np.mean(losses_policy), np.mean(losses_entropy)], ids=[0, 2, 3, 4])

        t1 = time.time()
        self._save_model(network=self.policy, path=self.path_final_model_policy)
        self._save_model(network=self.critic, path=self.path_final_model_critic)

        self.data.close()
        print("Done: %f" % (t1 - t0))

    def validate_custom(self, episode=None, episodes_total=None):
        validation_reward = self.validate(episode, episodes_total)
        if(self._save_model_for_each_validation):
            self._save_model(network=self.critic, path=os.path.join(self.path_models_folder, 'C%d.pt' % (episode + 1)))
        self._save_model(network=self.critic, path=self.path_final_model_critic)
        return validation_reward
