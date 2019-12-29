#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ENTMCdynA
#BSUB -n 6
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=2GB]"
#BSUB -u s144328@student.dtu.dk
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.6.2
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEntMC6/300_05" --max_episodes=100000 --num_envs=6 --entropy_weight=1.00 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEntMC6/300_06" --max_episodes=100000 --num_envs=6 --entropy_weight=1.00 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEntMC6/300_07" --max_episodes=100000 --num_envs=6 --entropy_weight=1.00 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEntMC6/300_08" --max_episodes=100000 --num_envs=6 --entropy_weight=1.00 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
