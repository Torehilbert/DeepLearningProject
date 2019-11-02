import torch
import numpy as np

from TimeManager import TimeManager as TM
from StatisticsContainer import MyData as MD


class ReinforcementTrainer:
    def __init__(self, policyNetwork, environment, optimizer):
        self.policy = policyNetwork
        self.env = environment
        self.optimizer = optimizer
        self.rollout_limit = 500
        self.discount_factor = 1.0

    def compute_returns(self, rewards, discount_factor):
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + discount_factor*returns[t+1]
        return returns

    def train(self, num_episodes=10000, val_freq=100):
        print('Starting training...')
        timer = TM(2).start()
        self.data = MD(3, maxsizes=[num_episodes, num_episodes, num_episodes//val_freq])
        for i in range(num_episodes):
            # simulate episode
            rollout = []
            s = self.env.reset()
            timer.start_timer(0)
            for j in range(self.rollout_limit):
                a, _ = self.policy.get_action(state=s, explore=True)
                s1, r, done, _ = self.env.step(a)
                rollout.append((s, a, r))
                s = s1
                if done:
                    break
            timer.stop_timer(0)

            # prepare batch
            rollout = np.array(rollout)
            states = np.vstack(rollout[:, 0])
            actions = np.vstack(rollout[:, 1])
            rewards = np.array(rollout[:, 2], dtype=float)
            returns = self.compute_returns(rewards, self.discount_factor)

            # policy gradient update
            self.optimizer.zero_grad()
            a_probs = self.policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
            loss = self.policy.loss(a_probs, torch.from_numpy(returns).float())
            loss.backward()
            self.optimizer.step()

            # bookkeeping
            self.data.add_data(sum(rewards), 0)
            self.data.add_data(loss.item(), 1)

            # print
            if (i+1) % val_freq == 0:
                timer.start_timer(1)
                validation_reward = self.validate(run_count=10)
                self.data.add_data(validation_reward, 2)
                print('{:5d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(self.data.extract_recent(0, val_freq)), validation_reward, np.mean(self.data.extract_recent(1, val_freq))))
                timer.stop_timer(1)
        timeElapsed = timer.stop()
        print('Finished training - Time elapsed: %d:%d:%d (Sim=%d%%, Val=%d%%)' % (timeElapsed[0], timeElapsed[1], timeElapsed[2], int(100*timer.fractions[0]), int(100*timer.fractions[1])))

    def validate(self, run_count=10):
        self.policy.eval()
        validation_rewards = []
        for _ in range(run_count):
            s = self.env.reset()
            reward = 0
            for _ in range(self.rollout_limit):
                with torch.no_grad():
                    a = self.policy(torch.from_numpy(np.atleast_2d(s)).float()).argmax().item()
                s, r, done, _ = self.env.step(a)
                reward += r
                if done: 
                    break
            validation_rewards.append(reward)
        return np.mean(validation_rewards)