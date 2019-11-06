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
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        self.network = network
        self.env = environment
        self.optimizer = optimizer
        self.rollout_limit = rollout_limit
        self.discount_factor = discount_factor
        self.model_path_save = model_path_save

        self.data_buffer_train = Data()
        self.data_buffer_eval = Data()

        self._save_model_counter = 0

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

    def validate(self, count=10):
        self.network.eval()
        with torch.no_grad():
            rollouts = [self.simulate_rollout(self.rollout_limit, False) for i in range(count)]

        total_rewards = []
        for i in range(len(rollouts)):
            total_rewards.append(np.sum(np.array(rollouts[i])[:, 2]))
        self.data_buffer_eval.add_data(total_rewards)
        self._save_model()
        return np.mean(total_rewards)

    def _save_model(self):
        if(self.model_path_save is not None):
            actual_path = self.model_path_save.split('.')[0] + "_%d"%self._save_model_counter + self.model_path_save.split('.')[1]
            self._save_model_counter += 1
            torch.save(self.network.state_dict(), actual_path)


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


class A2CTrainer(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        super(A2CTrainer, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, model_path_save)

    def train(self, episodes=100, val_freq=10, weight_actor=0.5, weight_entropy=0.001):
        print("Training started...")
        t0 = time.time()
        for episode in range(episodes):
            self.network.train()

            log_probs = []
            entropies = []
            values = []
            actions = []
            rewards = []

            state = self.env.reset()
            for j in range(self.rollout_limit):
                policy_dist, value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                action = (random.random() < np.cumsum(np.squeeze(policy_dist.detach().numpy()))).argmax()
                state_next, r, done, _ = self.env.step(action)

                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -torch.sum(policy_dist * torch.log(policy_dist))

                log_probs.append(log_prob)
                entropies.append(entropy)

                values.append(value)
                actions.append(actions)
                rewards.append(r)
                state = state_next

                if(done):
                    break

            qvals = np.zeros(len(rewards))
            qval = rewards[-1]
            for t in reversed(range(len(rewards))):
                qval = rewards[t] + self.discount_factor * qval
                qvals[t] = qval

            values = torch.stack(values)
            qvals = torch.FloatTensor(qvals)
            log_probs = torch.stack(log_probs)

            advantage = qvals - values
            actor_loss = - weight_actor * (log_probs * advantage.detach()).mean()
            critic_loss = (1 - weight_actor) * 0.5 * (advantage * advantage).mean()
            entropy_loss = weight_entropy * torch.mean(torch.stack(entropies))
            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.data_buffer_train.add_data([np.sum(rewards)])
            if((episode + 1) % val_freq == 0):
                validation_reward = self.validate()
                print(' %d%% : Validation reward: %d' % ((100 * (episode + 1) / episodes), validation_reward))

        self._save_model()
        t1 = time.time()
        print("Training ended: %f"%(t1-t0))


class A2CTrainerTorchify(BaseTrainer):
    def __init__(self, network, environment, optimizer, rollout_limit, discount_factor, model_path_save=None):
        super(A2CTrainerTorchify, self).__init__(network, environment, optimizer, rollout_limit, discount_factor, model_path_save)

    def train(self, episodes=100, val_freq=10, weight_actor=0.5, weight_entropy=0.001):
        print("Training started...")
        t0 = time.time()
        n_outputs = self.env.action_space.n
        for episode in range(episodes):
            self.network.train()

            actions = []
            prob_dist = []
            values = []
            rewards = []

            state = self.env.reset()
            for j in range(self.rollout_limit):
                policy_dist, value = self.network(torch.from_numpy(np.atleast_2d(state)).float())
                action = (random.random() < np.cumsum(np.squeeze(policy_dist.detach().numpy()))).argmax()
                state_next, r, done, _ = self.env.step(action)

                actions.append(action)
                prob_dist.append(policy_dist)

                values.append(value)
                
                rewards.append(r)
                state = state_next

                if(done):
                    break
            
            actions = torch.Tensor([actions]).to(torch.int64).view(-1, 1)
            prob_dist = torch.stack(prob_dist, dim=1).squeeze(0)
            values = torch.stack(values)

            log_prob_dist = torch.log(prob_dist)
            log_probs = log_prob_dist.gather(1, actions).view(-1)     
            entropy_mean = torch.mean(prob_dist * log_prob_dist)
  
            qvals = np.zeros(len(rewards))
            qval = rewards[-1]
            for t in reversed(range(len(rewards))):
                qval = rewards[t] + self.discount_factor * qval
                qvals[t] = qval
   
            qvals = torch.FloatTensor(qvals)

            advantage = qvals - values
            actor_loss = - weight_actor * (log_probs * advantage.detach()).mean()
            critic_loss = (1 - weight_actor) * 0.5 * (advantage * advantage).mean()
            entropy_loss = weight_entropy * entropy_mean
            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.data_buffer_train.add_data([np.sum(rewards)])
            if((episode + 1) % val_freq == 0):
                validation_reward = self.validate()
                print(' %d%% : Validation reward: %d' % ((100 * (episode + 1) / episodes), validation_reward))

        self._save_model()
        t1 = time.time()
        print("Training ended: %f"%(t1-t0))