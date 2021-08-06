#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m3670
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 03:00:00
#SBATCH -J train
#SBATCH -o logs/%x-%j.out

# Setup software
module load cgpu
module load tensorflow/gpu-1.15.0-py37

# Run the training
srun -l -u python train_tf_robbie.py $@