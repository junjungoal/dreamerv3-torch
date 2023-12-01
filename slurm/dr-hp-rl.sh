#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=140:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --partition=small
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=2
#SBATCH --exclude=dgk725

## output files
#SBATCH --output=/jmain02/home/J2AD008/wga37/mxr40-wga37/logs/%x.%j.out
#SBATCH --error=/jmain02/home/J2AD008/wga37/mxr40-wga37/logs/%x.%j.err

module purge
module load cuda/11.2
module load python/anaconda3

source ~/.bashrc
conda activate diffusion
module load pytorch/1.12.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export WANDB_API_KEY=833a88a1139a7f5523d4b39cfd80bbe8f6abb710
export WANDB_MODE=offline

SEED=${SLURM_ARRAY_TASK_ID}
TASK=Hopper-v3
GROUP=dreamer_hopper

cd /jmain02/home/J2AD008/wga37/mxr40-wga37/github/dreamerv3-torch
WANDB__SERVICE_WAIT=300 python3 dreamer.py --steps 1e6 --configs gym_proprio --task gym_$TASK --logdir /jmain02/home/J2AD008/wga37/mxr40-wga37/logdir/dreamerv3/$TASK/$GROUP/$SEED --seed $SEED --group $GROUP
