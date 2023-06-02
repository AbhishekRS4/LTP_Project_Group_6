#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64GB

# Use scratch due to limited space on /home
# export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

# Copy repo to local
cp -r $HOME/LTP_Project_Group_6/ $TMPDIR

module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

source $HOME/envs/env_lang_tech/bin/activate

cd $TMPDIR/LTP_Project_Group_6

which python3
python3 task_1/train_model.py \
--model google/flan-t5-base \
--data_path datasets/touche23_single_shot_prompt \
--run_name T5_base \
--checkpoint_save_path /scratch/$USER/models/ \
--learning_rate 1e-4 \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_epochs 30 \
--log_every_n_steps 20 \
--val_check_interval 1.0 \
--limit_val_batches 32 \
--force_cpu 0 \
--num_workers 16 \
--prompt_mode single_shot

deactivate
