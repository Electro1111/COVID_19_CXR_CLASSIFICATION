#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m3670
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 04:00:00
#SBATCH -J create_mod_COVIDx
#SBATCH -o logs/%x-%j.out

# Setup software
module load cgpu
module load pytorch/v1.5.1-gpu

# Run the training
srun -l -u python create_mod_COVIDx.py
