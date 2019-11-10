import torch
import torch.nn as nn
import numpy as np
import os
import random 
import time

from TimeManager import TimeManager as TM
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


class A2CTrainerTD(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        super(A2CTrainerTD, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, model_path_save)

    def train(self, episodes=100, val_freq=10):
        print("Training started...")
        num_outputs = self.env.action_space.n

        for i in range(episodes):
            reward_sum = 0
            state = self.env.reset()
            fading_factor = 1
            for j in range(self.rollout_limit):
                policy_dist, value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                value_detached = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy()

                action = np.random.choice(num_outputs, p=np.squeeze(dist))
                state_next, r, done, _ = self.env.step(action)

                if(not done):
                    _, value_next = self.network(torch.from_numpy(np.atleast_2d(state_next)).float())
                    value_next.detach()
                else:
                    value_next = 0

                delta = r + self.discount_factor * value_next - value

                log_prob = torch.log(policy_dist.squeeze(0)[action])
                critic_loss = delta * delta
                actor_loss = - fading_factor * delta.detach() * log_prob
                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                fading_factor = self.discount_factor * fading_factor
                state = state_next

                reward_sum += r

                if(done):
                    break

            self.data_buffer_train.add_data([reward_sum])
            if((i + 1) % val_freq == 0):
                validation_reward = self.validate()
                print(' %d%% : Validation reward: %d' % ((100 * (i + 1) / episodes), validation_reward))

        self._save_model()
        print("Training ended")


class A2CTrainerNG(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, path_output=None):
        super(A2CTrainerNG, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, path_output)
        self.data_buffer_loss = Data()
        self.data_buffer_actor_loss = Data()
        self.data_buffer_critic_loss = Data()
        self.data_buffer_entropy_loss = Data()
        self.data_buffer_gradient_norm = Data()

    def train(self, episodes=100, val_freq=10):

        for episode in range(episodes):
            log_probs = []
            values = []
            rewards = []
            actions = []
            masks = []
            entropies = []

            state = self.env.reset()
            self.network.train()
            for step in range(self.rollout_limit):
                # Evaluate network
                policy_dist, value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                values.append(value)

                # action
                action = (random.random() < np.cumsum(np.squeeze(policy_dist.detach().numpy()))).argmax()
                actions.append(action)

                # Take step
                state_next, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                masks.append(1.0 - done)

                # Entropy
                entropy = -(policy_dist.log() * policy_dist).sum()
                entropies.append(entropy)

                # Log probabilities
                log_prob = policy_dist[0, action].log()
                log_probs.append(log_prob)

                state = state_next
                if(done):
                    break

            with torch.no_grad():
                _, next_value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                returns = self.compute_returns(next_value.numpy(), rewards, masks, self.discount_factor)
                returns = torch.FloatTensor(returns)

            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            entropies = torch.stack(entropies)

            advantages = returns - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            entropy_loss = -entropies.mean()
            loss = 1.0 * actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss  # should be parameterized

            self.optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            self.data_buffer_train.add_data([np.sum(rewards)])
            self.data_buffer_loss.add_data([loss.detach().numpy()])
            self.data_buffer_actor_loss.add_data([actor_loss.detach().numpy()])
            self.data_buffer_critic_loss.add_data([critic_loss.detach().numpy()])
            self.data_buffer_entropy_loss.add_data([entropy_loss.detach().numpy()])
            self.data_buffer_gradient_norm.add_data([gnorm])

            if((episode + 1) % val_freq == 0):
                validation_reward = self.validate(index=(episode + 1))
                print(' %d%% : Validation reward: %d' % ((100 * (episode + 1) / episodes), validation_reward))

        self._save_model()
        self._save_info()
        print("Done")

    def compute_returns(self, next_value, rewards, masks, gamma):
        r = next_value
        returns = [0] * len(rewards)
        for step in reversed(range(len(rewards))):
            r = rewards[step] + gamma * r * masks[step]
            returns[step] = r
        return returns

class A2CTrainerStable(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, path_output=None):
        super(A2CTrainerStable, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, path_output)
        self.data_buffer_loss = Data()
        self.data_buffer_actor_loss = Data()
        self.data_buffer_critic_loss = Data()
        self.data_buffer_entropy_loss = Data()
        self.data_buffer_gradient_norm = Data()

    def train(self, episodes=100, val_freq=10):

        for episode in range(episodes):
            log_probs = []
            values = []
            rewards = []
            actions = []
            masks = []
            entropies = []

            state = self.env.reset()
            self.network.train()
            for step in range(self.rollout_limit):
                # Evaluate network
                policy_dist, value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                values.append(value)

                # action
                action = (random.random() < np.cumsum(np.squeeze(policy_dist.detach().numpy()))).argmax()
                actions.append(action)

                # Take step
                state_next, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                masks.append(1.0 - done)

                # Entropy
                entropy = -(policy_dist.log() * policy_dist).sum()
                entropies.append(entropy)

                # Log probabilities
                log_prob = policy_dist[0, action].log()
                log_probs.append(log_prob)

                state = state_next
                if(done):
                    break

            with torch.no_grad():
                _, next_value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                returns = self.compute_returns(next_value.numpy(), rewards, masks, self.discount_factor)
                returns = torch.FloatTensor(returns)

            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            entropies = torch.stack(entropies)

            advantages = returns - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            entropy_loss = -entropies.mean()
            loss = 1.0 * actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss  # should be parameterized

            self.optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            self.data_buffer_train.add_data([np.sum(rewards)])
            self.data_buffer_loss.add_data([loss.detach().numpy()])
            self.data_buffer_actor_loss.add_data([actor_loss.detach().numpy()])
            self.data_buffer_critic_loss.add_data([critic_loss.detach().numpy()])
            self.data_buffer_entropy_loss.add_data([entropy_loss.detach().numpy()])
            self.data_buffer_gradient_norm.add_data([gnorm])

            if((episode + 1) % val_freq == 0):
                validation_reward = self.validate(index=(episode + 1))
                print(' %d%% : Validation reward: %d' % ((100 * (episode + 1) / episodes), validation_reward))

        self._save_model()
        self._save_info()
        print("Done")

    def compute_returns(self, next_value, rewards, masks, gamma):
        r = next_value
        returns = [0] * len(rewards)
        for step in reversed(range(len(rewards))):
            r = rewards[step] + gamma * r * masks[step]
            returns[step] = r
        returns = np.array(returns)
        return (returns - np.mean(returns)) / np.std(returns)
