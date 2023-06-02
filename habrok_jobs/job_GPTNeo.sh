#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

# Copy git repo to local
# cp -r ~/LTP_Project_Group_6/ $TMPDIR

module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

source $HOME/envs/env_lang_tech/bin/activate

cd $HOME/LTP_Project_Group_6

which python3
python3 task_1/train_model.py \
--model EleutherAI/pythia-410m-deduped \
--data_path datasets/touche23_neo_single_shot_prompt \
--run_name Test_GPTNeo \
--checkpoint_save_path /scratch/$USER/models/ \
--learning_rate 1e-4 \
--train_batch_size 16 \
--eval_batch_size 16 \
--max_epochs 15 \
--log_every_n_steps 20 \
--val_check_interval 1.0 \
--limit_val_batches 15 \
--force_cpu 0 \
--prompt_mode few_shot \
--neo_mode 1


