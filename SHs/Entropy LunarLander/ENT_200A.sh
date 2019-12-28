#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ENT_DynA
#BSUB -n 24
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=2GB]"
#BSUB -u s144328@student.dtu.dk
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.6.2
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEnt/400_01" --max_episodes=25000 --num_envs=23 --entropy_weight=1.0 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEnt/400_02" --max_episodes=25000 --num_envs=23 --entropy_weight=1.0 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEnt/400_03" --max_episodes=25000 --num_envs=23 --entropy_weight=1.0 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --output_path="OutputEnt/400_04" --max_episodes=25000 --num_envs=23 --entropy_weight=1.0 --entropy_weight_end=0.01 --hiddensize=128 --rollout_limit=500 --num_steps=10
