#!/bin/bash
#SBATCH --account=engs-a2i
#SBATCH --nodes=1
#SBATCH --time=024:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --partition=short
#SBATCH --mem=20G
#SBATCH --reservation=a2i082023
## output files
#SBATCH --output=/data/engs-a2i/mans4401/projects/dreamerv3-torch/slurm_logs/%x.%j.out
#SBATCH --error=/data/engs-a2i/mans4401/projects/dreamerv3-torch/slurm_logs/%x.%j.err
source ~/.zshrc
conda activate diffusion-wm

SEED=${SLURM_ARRAY_TASK_ID}
TASK=$1
GROUP=$2
H=32

WANDB__SERVICE_WAIT=300 python3 dreamer.py --configs dmc_proprio --task dmc_$TASK --logdir ./logdir/$TASK/$GROUP/$SEED --seed $SEED --group $GROUP-$TASK --eval_batch_length $H --imag_horizon $H
