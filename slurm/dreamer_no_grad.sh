#!/bin/bash
#SBATCH --account=engs-a2i
#SBATCH --nodes=1
#SBATCH --time=024:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --partition=short
#SBATCH--reservation=a2i202310
#SBATCH --mem=20G
## output files
#SBATCH --output=/data/engs-a2i/mans4401/projects/dreamerv3-torch/slurm_logs/%x.%j.out
#SBATCH --error=/data/engs-a2i/mans4401/projects/dreamerv3-torch/slurm_logs/%x.%j.err
source ~/.zshrc
conda activate diffusion-wm

SEED=${SLURM_ARRAY_TASK_ID}
TASK=$1
GROUP=$2

export MUJOCO_GL='osmesa'
WANDB__SERVICE_WAIT=300 python3 dreamer.py --configs gym_proprio_reinforce_no_grad --task gym_$TASK --logdir ./logdir/$TASK/$GROUP/$SEED --seed $SEED --group $GROUP-$TASK
