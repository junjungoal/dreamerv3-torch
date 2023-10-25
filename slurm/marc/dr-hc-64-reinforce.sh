#!/bin/bash
#SBATCH --account=engs-a2i
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:rtx:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-1
#SBATCH --partition=short
#SBATCH --mem=16G 
#SBATCH --reservation=a2i202310

## output files
#SBATCH --output=/home/pemb5572/logs/%x.%j.out
#SBATCH --error=/home/pemb5572/logs/%x.%j.err

module purge
module load CUDA/11.7.0
module load Anaconda3/2022.05

source ~/.bashrc
cd /home/pemb5572/github/dreamerv3-torch
conda activate diffusion

SEED=${SLURM_ARRAY_TASK_ID}
TASK=HalfCheetah-v3
GROUP=dreamer-long-horizon-reinforce
H=64

WANDB__SERVICE_WAIT=300 python3 dreamer.py --configs gym_proprio_reinforce_no_grad --task gym_$TASK --logdir /data/engs-a2i/pemb5572/dreamerv3/$TASK/$GROUP/$H/$SEED --seed $SEED --group $GROUP --eval_batch_length $H --imag_horizon $H
