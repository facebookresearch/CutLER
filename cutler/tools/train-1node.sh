# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/bin/bash
#SBATCH -p devlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=512G
#SBATCH --time 2000
#SBATCH -o "submitit/slurm-%j.out"

srun tools/single-node_run.sh $@