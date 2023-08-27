#!/bin/bash
#SBATCH -p learnfair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time 10000
#SBATCH -o "submitit/videocutler/slurm-%j.out"

srun single-node-video_run.sh $@