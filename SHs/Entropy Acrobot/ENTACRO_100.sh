#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Acrobot100
#BSUB -n 7
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=2GB]"
#BSUB -u s144328@student.dtu.dk
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.6.2
python3 ~/DLProject/_A3C/main_a3c.py --env="Acrobot-v1" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="Output/100_5" --max_episodes=5000 --num_envs=6 --entropy_weight=1.0 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="Acrobot-v1" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="Output/100_6" --max_episodes=5000 --num_envs=6 --entropy_weight=1.0 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="Acrobot-v1" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="Output/100_7" --max_episodes=5000 --num_envs=6 --entropy_weight=1.0 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="Acrobot-v1" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="Output/100_8" --max_episodes=5000 --num_envs=6 --entropy_weight=1.0 --hiddensize=128 --rollout_limit=500 --num_steps=10
