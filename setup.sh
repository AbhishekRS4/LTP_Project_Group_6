module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

mkdir envs
cd envs
python3 -m venv env_lang_tech
pip install -r LTP_Project_Group_6/requirements.txt

