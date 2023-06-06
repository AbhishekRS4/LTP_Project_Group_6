# Habrok instructions

## Virtual env setup
* Remove `torch==1.13.1` from `requirements.txt` because we can load pre-installed PyTorch.
* In the below set of commands, **env_lang_tech** is the name of the virtual env.
```
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
python3 -m venv env_lang_tech
pip install -r LTP_Project_Group_6/requirements.txt
```

## Submitting job
* Add the following in a bash script for example `job.sh`
```
#!/bin/bash
#SBATCH --time=23:55:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000

module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

source /home4/s0000000/language_technology/env_lang_tech/bin/activate

which python3
python3 ./LTP_Project_Group_6/task_1/train_model.py
```
* Submit a job
```
sbatch job.sh
```

## To copy files
* From local system to server
```
rsync -aP dataset_files.tar s0000000@login1.hb.hpc.rug.nl:/projects/s0000000/
```
* First copy any files to `/projects/s0000000/` since this is a permanent storage but cannot be accessed by the cluster
* Copy files needed by the scripts to `/scratch/s0000000/`
* After this, load the files in the scripts from the appropriate paths in `/scratch/s0000000/`

## Commands for Jeroen

### Login
ssh s3416402@login1.hb.hpc.rug.nl

### Copy repo
rsync -r LTP_Project_Group_6 s3416402@login1.hb.hpc.rug.nl:/home3/s3416402/

rsync -r artifacts/T5_large_augmented_single_shot_0603-22:29:41 s3416402@login1.hb.hpc.rug.nl:/home3/s3416402/LTP_Project_Group_6/artifacts

rsync -r test_T5_large.sh s3416402@login1.hb.hpc.rug.nl:/home3/s3416402/LTP_Project_Group_6

rsync -r s3416402@login1.hb.hpc.rug.nl:/scratch/s3416402/models/test_lower_loss artifacts/

rsync -r s3416402@login1.hb.hpc.rug.nl:/scratch/s3416402/models/0604-15:26:47-touche23-epoch=49-val/f1=0.79.ckpt artifacts/T5_base_few_shot_0604-15:26:47



### Interactive command line
srun --partition=gpushort --gpus-per-node=a100:1 --mem=64GB --time=3:00:00 --job-name=test --pty /bin/bash

### Check GPU availability
squeue | grep gpu

### Job info
squeue -u $USER

### Check disk space
hbquota


