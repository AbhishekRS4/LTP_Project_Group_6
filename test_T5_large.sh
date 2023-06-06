#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64GB

# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

# Copy repo to local
# cp -r $HOME/LTP_Project_Group_6/ $TMPDIR

module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

source $HOME/envs/env_lang_tech/bin/activate

cd $HOME/LTP_Project_Group_6

which python3
python task_1/test.py \
--results_save_file flan_T5_large_single_shot_augmented \
--model_type google/flan-t5-base \
--dataset_path datasets/touche23_single_shot_prompt \
--file_path_model /scratch/public/tmp/model.ckpt \
--longT5_mode 0 \
--eval_batch_size 16


deactivate
