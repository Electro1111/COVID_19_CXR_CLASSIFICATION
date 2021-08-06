#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m3670
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 01:00:00
#SBATCH -J _pytorch_class_
#SBATCH -o logs/%x-%j.out

# Setup software
module load cgpu
module load pytorch/v1.5.1-gpu

# Run the training
srun -l -u python robbie_pytorch_train.py $@