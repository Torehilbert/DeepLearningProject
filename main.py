import torch
import torch.optim as optim
import gym
import threading
import time
import os

from TrainVisualizer import TrainTracker
from network_architectures import PolicyNet, PolicyNetDouble
from ReinforcementTrainer import ReinforcementTrainer as RLTrainer


if __name__ == "__main__":
    output_folder = r"C:\Users\ToreH\OneDrive - Københavns Universitet\Skole\02456 Deep Learning\Project\DeepLearningProject\output"
    
    load_ini_model = True
    save_final_model = True
    final_model_path = r"C:\Source\DeepLearningProject\IniModel6464\model_final.pt"
    ini_model_path = final_model_path # r"C:\Source\DeepLearningProject\IniModel6464\iniModel5000.pt"

    # Testing Net
    environmentName = 'LunarLander-v2'  #'CartPole-v0'  #
    hiddenSizes = [64,64]

    env = gym.make(environmentName)

    nInputs = env.observation_space.shape[0]
    nActions = env.action_space.n

    # training settings
    batch_size = 10         # hyper 1 parameter #10 is good
    num_episodes = (5000//batch_size)*batch_size
    val_freq = ((num_episodes//10)//batch_size)*batch_size

    lr_policy = 0.002        # hyper 2 parameter #0.05 is good ()
    use_baseline = True     # hyper 3 parameter

    rollout_limit = 500  # max rollout length
    discount_factor = 1.0  # reward discount factor (gamma), 1.0 = no discount


    # setup policy network
    #policy = PolicyNet(nInputs, hiddenSize, nActions)
    policy = PolicyNetDouble(nInputs, hiddenSizes, nActions)
    if(load_ini_model):
        policy.load_state_dict(torch.load(ini_model_path))

    optimizer = optim.Adam(policy.parameters(), lr=lr_policy)

    # train policy network
    trainer = RLTrainer(policy, env, optimizer)
    trainer.rollout_limit = rollout_limit
    trainer.discount_factor = discount_factor

    thread = threading.Thread(target=trainer.train, kwargs={'num_episodes':num_episodes, 'batch_size':batch_size, 'use_baseline':use_baseline, 'model_output_folder':output_folder})
    thread.start()
    time.sleep(1)

    # track training
    outfiles = [os.path.join(output_folder, 'train_reward.csv'), 
                os.path.join(output_folder, 'validation_reward.csv')]

    tv = TrainTracker([trainer.dataTrain, trainer.dataEval], thread, smooth_alphas=[0.05, 1], out_filepaths=outfiles)
    tv.initialize()
    tv.format(0, 'Iteration (training)', 'Training reward', [0, num_episodes])
    tv.format(1, 'Iteration (validation)', 'Validation reward', [0, num_episodes//val_freq])
    tv.start(update_interval=0.25)

    # save final model
    if(save_final_model):
        torch.save(policy.state_dict(), final_model_path)
