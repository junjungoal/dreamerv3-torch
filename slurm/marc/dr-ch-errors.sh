#!/bin/bash
#SBATCH --account=engs-a2i
#SBATCH --nodes=1
#SBATCH --time=48:00:00
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
GROUP=dreamer-errors-nov9_datasets
DATASET=/jmain02/home/J2AD008/wga37/mxr40-wga37/github/diffusion-wm/datasets/test_datasets_nov9/HalfCheetah
LOAD_STEP=1000000
H=200

WANDB__SERVICE_WAIT=300 python3 train_wm_only.py --configs gym_proprio --task gym_$TASK --logdir /data/engs-a2i/pemb5572/dreamerv3/$TASK/$GROUP/$H/$SEED --seed $SEED --group $GROUP --eval_batch_length $H --load_path $DATASET --load_step $LOAD_STEP 