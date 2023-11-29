#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --partition=small
#SBATCH --mem=16G 
#SBATCH --cpus-per-task=4
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
GROUP=errors-dreamer-v3-nov12_datasets
DATASET=/jmain02/home/J2AD008/wga37/mxr40-wga37/github/diffusion-wm/datasets/final_datasets_nov12/final-rl-runs-lowtrainratio_seed1_Hopper
LOAD_STEP=1000000
H=300

cd /jmain02/home/J2AD008/wga37/mxr40-wga37/github/dreamerv3-torch
/jmain02/home/J2AD008/wga37/mxr40-wga37/.conda/envs/diffusion/bin/python3 train_wm_only.py --configs gym_proprio --task gym_$TASK --logdir /data/engs-a2i/pemb5572/dreamerv3/$TASK/$GROUP/$H/$SEED --seed $SEED --group $GROUP --eval_batch_length $H --load_path $DATASET --load_step $LOAD_STEP 
