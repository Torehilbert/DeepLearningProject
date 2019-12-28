#!/bin/sh
#BSUB -q gpuv100
#BSUB -J DFLL098B
#BSUB -n 24
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=2GB]"
#BSUB -u s144328@student.dtu.dk
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.6.2
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --discount=0.98 --output_path="OutputDFLL/098_05" --max_episodes=40000 --num_envs=23 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --discount=0.98 --output_path="OutputDFLL/098_06" --max_episodes=40000 --num_envs=23 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --discount=0.98 --output_path="OutputDFLL/098_07" --max_episodes=40000 --num_envs=23 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
python3 ~/DLProject/_A3C/main_a3c.py --env="LunarLander-v2" --lr_policy=1e-4 --lr_critic=1e-3 --discount=0.98 --output_path="OutputDFLL/098_08" --max_episodes=40000 --num_envs=23 --entropy_weight=0.1 --hiddensize=128 --rollout_limit=500 --num_steps=10
