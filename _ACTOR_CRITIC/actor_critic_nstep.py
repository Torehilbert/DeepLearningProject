import copy
import gym
import threading
import torch
import torch.optim as optim
import numpy as np
import random
import sys

DEFAULT_TMAX = 10


class A3CTrainer:
    def __init__(self, policy_network, critic_network, environment_name, settings):
        self.policy = policy_network  # global policy
        self.critic = critic_network  # global critic
        self.environment_name = environment_name
        self.settings = settings  # args from argparse parser class

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)
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
        policy = copy.deepcopy(self.policy)     # local policy
        critic = copy.deepcopy(self.critic)     # local critic
        env = gym.make(self.environment_name)   # local environment

        done = True
        while(1):
            counter = 0
            if(done):
                state = env.reset()
                done = False

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

                states.append(state)
                rewards.append(reward)
                actions.append(action)
                policy_dists.append(policy_distribution)

                counter += 1
                state = state_next
                if(done):
                    break

            # 2) Gradients from experience
            policy_dists = torch.stack(policy_dists).squeeze()
            log_policy_dists = torch.log(policy_dists + 1e-9)
            actions = torch.from_numpy(np.array(actions)).long().view(-1, 1)
            log_prob = log_policy_dists.gather(1, actions)

            states = torch.stack(states).squeeze()

            value_next = critic(state).detach() if not done else 0
            returns = np.zeros(shape=(counter,))
            for t in reversed(range(counter)):
                value_next = rewards[t] + self.settings.discount * value_next
                returns[t] = value_next
            
            returns = torch.from_numpy(returns).float()
            values = critic(states).squeeze()
            advantages = returns - values

            critic.zero_grad()
            loss_critic = (advantages * advantages).sum()
            loss_critic.backward()

            policy.zero_grad()
            loss_policy = -(log_prob * advantages.detach()).sum()
            loss_policy.backward()

            # 3) Send gradients to global network
            self.update_global(policy_local=policy, critic_local=critic)
            sys.exit(1)

            print(advantages.size())
            print(returns.size())
            print(states.size())

     
            sys.exit(1)

            out = critic(states).squeeze()
            print("State size: ", state.size())
            print("States size: ", states.size())
            print("Out size: ", out.size())
            sys.exit(1)

            # Manage global training status
            self.Tlock.acquire()
            self.T += t
            exit_flag = 1 if self.T > self.settings.Tmax else 0
            self.Tlock.release()
            if(exit_flag):
                break

    def update_global(self, policy_local, critic_local):  # policy_params, critic_params):
        # 1) Transfer gradients
        policy_params = [p.grad for p in policy_local.parameters()]
        critic_params = [p.grad for p in critic_local.parameters()]
        self.lock_params.acquire()
        self.policy.zero_grad()
        idx_counter = 0
        for p in self.policy.parameters():
            p.grad = policy_params[idx_counter]
            idx_counter += 1
        
        self.critic.zero_grad()
        idx_counter = 0
        for p in self.critic.parameters():
            p.grad = critic_params[idx_counter]
            idx_counter += 1
        
        # 2) Optimizer step
        self.optimizer_policy.step()
        self.optimizer_critic.step()

        # 3) Transfer global weights to local
        policy_weights = [p.data for p in policy_local.parameters()]
        critic_weights = [p.data for p in critic_local.parameters()]
        
        idx_counter = 0
        for p in policy_local.parameters():
            p.data.copy_(policy_weights[idx_counter])
            idx_counter += 1

        idx_counter = 0
        for p in critic_local.parameters():
            p.data.copy_(critic_weights[idx_counter])
            idx_counter += 1
        
        self.lock_params.release()


if __name__ == "__main__":
    class DummySettings:
        def __init__(self):
            self.nsteps = 100
            self.nthreads = 1
            self.Tmax = 1000
            self.discount = 0.99
    
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
