import copy
import gym
import threading
import torch
import torch.optim as optim
import numpy as np
import random
import sys

from network_policy import Policy


DEFAULT_TMAX = 10


class A3CTrainer:
    def __init__(self, policy_network, critic_network, environment_name, settings):
        self.policy = policy_network  # global policy
        self.critic = critic_network  # global critic
        self.environment_name = environment_name
        self.settings = settings  # args from argparse parser class

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.optimizer_critic = optim.SGD(self.critic.parameters(), lr=1e-3)
        #   settings.nsteps
        #   settings.nthreads
        #   settings.Tmax

    def train(self):
        self.T = 0
        self.Tlock = threading.Lock()
        self.lock_params = threading.Lock()

        self.threads = [threading.Thread(target=self.thread_train) for t in range(self.settings.nthreads)]
        for t in self.threads:
            t.start()
            print('Thread started!')

        for t in self.threads:
            t.join()
        print('Done')

    def thread_train(self):
        policy = Policy(4, 2, 16)  # copy.deepcopy(self.policy)     # local policy
        critic = copy.deepcopy(self.critic)     # local critic
        env = gym.make(self.environment_name)   # local environment
        opt = optim.Adam(policy.parameters(), lr=1e-1)

        done = True
        while(1):
            counter = 0
            if(done):
                state = env.reset()
                total_reward = 0
                done = False
            #self._debug_print_network(policy)
            # 1) Acquire environment experience
            states = []
            rewards = []
            actions = []
            policy_dists = []
            for t in range(self.settings.nsteps):
                state = torch.from_numpy(np.atleast_2d(state)).float()
                policy_distribution = policy(state)
                action = (random.random() < np.cumsum(np.squeeze(policy_distribution.detach().numpy()))).argmax()
                state_next, reward, done, _ = env.step(action)
                state_next = torch.from_numpy(np.atleast_2d(state_next)).float()
                
                total_reward += reward
                states.append(state)
                rewards.append(reward)
                actions.append(action)
                policy_dists.append(policy_distribution)

                counter += 1
                state = state_next
                if(done):
                    print(total_reward)
                    break

            # 2) Gradients from experience
            policy_dists = torch.stack(policy_dists).squeeze(1)
            log_policy_dists = torch.log(policy_dists + 1e-9)
            actions = torch.from_numpy(np.array(actions)).long().view(-1, 1)
            log_prob = log_policy_dists.gather(1, actions)

            states = torch.stack(states).squeeze()

            #value_next = 0  # critic(state).detach() if not done else 0
            returns = np.zeros(shape=(len(rewards),))
            for t in reversed(range(len(rewards) - 1)):
                returns[t] = rewards[t] + self.settings.discount * returns[t + 1]
            #print(returns)
            returns = torch.from_numpy(returns).float()
            #values = critic(states).squeeze()
            #advantages = returns - values

            #critic.zero_grad()
            #loss_critic = (advantages * advantages).sum()
            #loss_critic.backward()

            opt.zero_grad()
            loss_policy = -(log_prob * returns.detach()).mean()
            loss_policy.backward()
            opt.step()

            # 3) Send gradients to global network
            #self.update_global(policy_local=policy, critic_local=critic)
            #print("\nAFTER")
            #self._debug_print_network(policy)
            #sys.exit()
            # Manage global training status
            self.Tlock.acquire()
            self.T += counter
            exit_flag = 1 if self.T > self.settings.Tmax else 0
            self.Tlock.release()
            if(exit_flag):
                break

    def update_global(self, policy_local, critic_local):  # policy_params, critic_params):
        self.optimizer_policy.zero_grad()
        for lp, gp in zip(policy_local.parameters(), self.policy.parameters()):
            gp._grad = lp.grad
        self.optimizer_policy.step()
        policy_local.load_state_dict(self.policy.state_dict())

        # # 1) Transfer gradients
        # policy_grads = [p.grad for p in policy_local.parameters()]
        # critic_grads = [p.grad for p in critic_local.parameters()]
        # self.lock_params.acquire()
        # self.policy.zero_grad()
        # idx_counter = 0
        # for p in self.policy.parameters():
        #     p.grad = policy_grads[idx_counter].detach()
        #     idx_counter += 1

        # self.critic.zero_grad()
        # idx_counter = 0
        # for p in self.critic.parameters():
        #     p.grad = critic_grads[idx_counter].detach()
        #     idx_counter += 1

        # # 2) Optimizer step
        # self.optimizer_policy.step()
        # self.optimizer_critic.step()
      
        # # 3) Transfer global weights to local
        # policy_local.load_state_dict(self.policy.state_dict())
        # critic_local.load_state_dict(self.critic.state_dict())
        # self.lock_params.release()

    def _debug_print_network(self, network):
        for p in network.parameters():
            print(p.data)


if __name__ == "__main__":
    class DummySettings:
        def __init__(self):
            self.nsteps = 1000
            self.nthreads = 1
            self.Tmax = 10000
            self.discount = 1.00

    from network_policy import Policy
    from network_critic import Critic
    env_name = 'CartPole-v0'
    env_dummy = gym.make(env_name)
    n_inputs = env_dummy.observation_space.shape[0]
    n_actions = env_dummy.action_space.n

    policy = Policy(n_inputs, n_actions, 4)
    critic = Critic(n_inputs, 4)
    settings = DummySettings()

    trainer = A3CTrainer(policy, critic, 'CartPole-v0', settings)
    trainer.train()
