#!/bin/sh
#BSUB -q gpuv100
#BSUB -J MountainCar_6_C
#BSUB -n 7
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=2GB]"
#BSUB -u s144328@student.dtu.dk
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.6.2
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputMC/06_09" --max_episodes=50000 --num_envs=6 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputMC/06_10" --max_episodes=50000 --num_envs=6 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputMC/06_11" --max_episodes=50000 --num_envs=6 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="MountainCar-v0" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputMC/06_12" --max_episodes=50000 --num_envs=6 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
