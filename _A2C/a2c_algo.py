import numpy as np
import random
import torch
import torch.optim as optim
import os
import time
import sys

shared_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_SHARED')
sys.path.append(shared_path)

from Data import Data


DEFAULT_LEARNING_RATE_POLICY = 1e-3
DEFAULT_LEARNING_RATE_CRITIC = 1e-4
DEFAULT_ENTROPY_WEIGHT = 0.01
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_MAX_GRAD_POLICY = 0.5
DEFAULT_MAX_GRAD_CRITIC = 0.5

DEFAULT_NUMBER_OF_EPISODES = 500
DEFAULT_ROLLOUT_LIMIT = 500
DEFAULT_VALIDATION_RATE = 50
DEFAULT_VALIDATION_COUNT = 10
DEFAULT_VALIDATION_PRINT = True
DEFAULT_OUTPUT_PATH = r"C:\Source\DeepLearningProject\Outputs\latest"


class A2CTrainer:
    def __init__(self, policy_network, critic_network, environment, **kwargs):
        self.env = environment

        self.policy = policy_network
        self.lr_policy = kwargs['lr_policy'] if 'lr_policy' in kwargs else DEFAULT_LEARNING_RATE_POLICY
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        self.max_grad_norm_policy = kwargs['max_grad_norm_policy'] if 'max_grad_norm_policy' in kwargs else DEFAULT_MAX_GRAD_POLICY
        self.entropy_weight = kwargs['entropy_weight'] if 'entropy_weight' in kwargs else DEFAULT_ENTROPY_WEIGHT
        self.critic = critic_network
        self.lr_critic = kwargs['lr_critic'] if 'lr_critic' in kwargs else DEFAULT_LEARNING_RATE_CRITIC
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.max_grad_norm_critic = kwargs['max_grad_norm_critic'] if 'max_grad_norm_critic' in kwargs else DEFAULT_MAX_GRAD_CRITIC

        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else DEFAULT_DISCOUNT_FACTOR
        self.number_of_episodes = kwargs['number_of_episodes'] if 'number_of_episodes' in kwargs else DEFAULT_NUMBER_OF_EPISODES
        self.rollout_limit = kwargs['rollout_limit'] if 'rollout_limit' in kwargs else DEFAULT_ROLLOUT_LIMIT
        self.validation_rate = kwargs['validation_rate'] if 'validation_rate' in kwargs else DEFAULT_VALIDATION_RATE
        self.validation_count = kwargs['validation_count'] if 'validation_count' in kwargs else DEFAULT_VALIDATION_COUNT
        self.validation_print = kwargs['validation_print'] if 'validation_print' in kwargs else DEFAULT_VALIDATION_PRINT

        self.path_output_folder = kwargs['path_output'] if 'path_output' in kwargs else DEFAULT_OUTPUT_PATH
        self.path_models_folder = os.path.join(self.path_output_folder, 'models')
        self.path_tracks_folder = os.path.join(self.path_output_folder, 'tracks')
        self.path_final_model_policy = os.path.join(self.path_output_folder, 'policy.pt')
        self.path_final_model_critic = os.path.join(self.path_output_folder, 'critic.pt')

        os.makedirs(self.path_output_folder, exist_ok=True)
        os.makedirs(self.path_models_folder, exist_ok=True)
        os.makedirs(self.path_tracks_folder, exist_ok=True)

        self.data_buffer_train = Data()
        self.data_buffer_eval = Data()
        self.data_buffer_loss_policy = Data()
        self.data_buffer_loss_critic = Data()
        self.data_buffer_loss_entropy = Data()

    def train(self, number_of_episodes=None, rollout_limit=None):
        t0 = time.time()
        number_of_episodes = self.number_of_episodes if number_of_episodes is None else number_of_episodes
        rollout_limit = self.rollout_limit if rollout_limit is None else rollout_limit

        for episode in range(number_of_episodes):
            actions = []
            rewards = []
            states = []
            pol_dists = []
            log_pol_dists = []

            state = self.env.reset()
            for t in range(rollout_limit):
                state_torch = torch.from_numpy(np.atleast_2d(state)).float()
                policy_distribution = self.policy(state_torch)

                action = (random.random() < np.cumsum(np.squeeze(policy_distribution.detach().numpy()))).argmax()
                state_next, reward, done, _ = self.env.step(action)

                log_policy_distribution = torch.log(policy_distribution + 1e-9)
                actions.append(action)
                states.append(state_torch)
                rewards.append(reward)
                pol_dists.append(policy_distribution)
                log_pol_dists.append(log_policy_distribution)

                state = state_next
                if(done):
                    break

            value_predictions = self.critic(torch.stack(states).squeeze()).squeeze()

            log_pol_dists = torch.stack(log_pol_dists).squeeze()
            pol_dists = torch.stack(pol_dists).squeeze()
            entropy = -2 * (log_pol_dists * pol_dists).mean()  # the 2 in front is compensation for mean actually dividing with double the true

            log_probs = log_pol_dists.gather(1, torch.from_numpy(np.array(actions)).long().unsqueeze(1)).squeeze()

            # compute returns
            returns = np.zeros(shape=(len(rewards), 1))
            R = 0 if done else self.critic(torch.from_numpy(np.atleast_2d(state)).float()).detach().numpy()
            for t in reversed(range(len(rewards))):
                R = rewards[t] + self.gamma * R
                returns[t] = R

            # compute advantages
            returns = torch.from_numpy(returns).float()
            advantages = returns - value_predictions

            # optimize
            loss_critic = self.optimize_critic(advantages)
            loss_policy, loss_entropy = self.optimize_policy(advantages, log_probs, entropy)

            # book-keeping
            self.data_buffer_train.add_data([np.sum(rewards)])
            self.data_buffer_loss_critic.add_data([loss_critic])
            self.data_buffer_loss_policy.add_data([loss_policy])
            self.data_buffer_loss_entropy.add_data([loss_entropy])

            # validation
            if((episode + 1) % self.validation_rate == 0):
                self.validate(episode, number_of_episodes)

        t1 = time.time()
        self._save_model()
        print("Done: %f" % (t1 - t0))

    def optimize_critic(self, advantages):
        loss_critic = (advantages * advantages).mean()
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.optimizer_critic.step()
        return loss_critic.detach().numpy()

    def optimize_policy(self, advantages, log_probs, entropies):
        advantages = advantages.detach()
        loss_policy = -(log_probs * advantages).mean()
        loss_entropy = -self.entropy_weight * entropies
        loss = loss_policy + loss_entropy
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm_policy)
        self.optimizer_policy.step()
        return loss_policy.detach().numpy(), loss_entropy.detach().numpy()

    def validate(self, episode=None, episodes_total=None):
        with torch.no_grad():
            results = []
            for rep in range(self.validation_count):
                state = self.env.reset()
                rewards = []
                for i in range(self.rollout_limit):
                    probs = self.policy(torch.from_numpy(np.atleast_2d(state)).float()).detach().numpy()
                    action = np.squeeze(probs).argmax()
                    state, reward, done, _ = self.env.step(action)
                    rewards.append(reward)
                    if done:
                        break
                results.append(np.sum(rewards))

            validation_reward = np.mean(results)
            self.data_buffer_eval.add_data([validation_reward])
            if(self.validation_print):
                if(episode is not None and episodes_total is None):
                    print("%d.  Validation reward: %.3f" % (episode + 1, validation_reward))
                elif(episode is not None):
                    print("%d%%. Validation reward: %.3f" % (100 * (episode + 1) / episodes_total, validation_reward))
                else:
                    print('Validation reward: %.3f' % validation_reward)

            self._save_model(paths=[os.path.join(self.path_models_folder, 'P%d.pt' % (episode + 1)), os.path.join(self.path_models_folder, 'C%d.pt' % (episode + 1))])
            self._save_model()

    def _save_model(self, paths=None):
        policy_dict = self.policy.state_dict()
        critic_dict = self.critic.state_dict()

        if(paths is not None):
            torch.save(policy_dict, paths[0])
            torch.save(critic_dict, paths[1])
        else:
            torch.save(policy_dict, self.path_final_model_policy)
            torch.save(critic_dict, self.path_final_model_critic)
