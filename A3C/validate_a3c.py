import time
import copy
import gym
import torch
import numpy as np
import os


def validate(policy_global, critic_global, steps_count, episode_count, args):
    env = gym.make(args.env)
    policy = copy.deepcopy(policy_global)

    if(args.output_path is not None):
        os.makedirs(args.output_path, exist_ok=True)
        log = open(os.path.join(args.output_path, 'log.csv'), 'w')
        log.write('Steps,Episodes,Time,Reward\n')
        log.close()
        path_policy = os.path.join(args.output_path, 'policy.pt')
        path_critic = os.path.join(args.output_path, 'critic.pt')
        info = open(os.path.join(args.output_path, 'info.txt'), 'w')
        for key, value in vars(args).items():
            info.write(key + " = " + str(value) + "\n")
        info.close()

    t0 = time.time()
    while(1):
        current_time = time.time() - t0
        current_step = steps_count.value
        current_episode = episode_count.value
        policy.load_state_dict(policy_global.state_dict())

        all_epslen = []
        all_reward_sums = []
        for rep in range(args.validation_count):
            state = env.reset()
            epslen = 0
            reward_sum = 0
            for t in range(args.rollout_limit):
                state = torch.from_numpy(np.atleast_2d(state)).float()
                policy_distribution = policy(state)
                action = policy_distribution.detach().numpy().argmax()
                state_next, reward, done, _ = env.step(action)

                reward_sum += reward
                epslen += 1
                state = state_next

                if(done):
                    break

            all_epslen.append(epslen)
            all_reward_sums.append(reward_sum)

        if(args.output_path is not None):
            log = open(os.path.join(args.output_path, 'log.csv'), 'a')
            log.write(str(current_step) + "," + str(current_episode) + "," + str(current_time) + "," + str(np.mean(all_reward_sums)) + "\n")
            log.close()
            torch.save(policy_global.state_dict(), path_policy)
            torch.save(critic_global.state_dict(), path_critic)
        print("%5d: Validation reward: %7.0f    Episode length: %6d" % (current_episode, np.mean(all_reward_sums), np.mean(all_epslen)))

        if(args.max_steps is not None and current_step >= args.max_steps):
            break

        if(args.max_episodes is not None and current_episode >= args.max_episodes):
            break


def render(policy_global, args):
    env = gym.make(args.env)
    policy = copy.deepcopy(policy_global)

    while(1):
        policy.load_state_dict(policy_global.state_dict())
        state = env.reset()
        for t in range(args.rollout_limit):
            state = torch.from_numpy(np.atleast_2d(state)).float()
            policy_distribution = policy(state)
            action = policy_distribution.detach().numpy().argmax()
            state_next, reward, done, _ = env.step(action)
            env.render()

            state = state_next
            if(done):
                break
        
        time.sleep(args.render_pause)
