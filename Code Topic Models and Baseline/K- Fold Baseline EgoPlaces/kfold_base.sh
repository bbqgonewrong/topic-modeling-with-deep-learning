#!/bin/bash
#SBATCH --job-name='base_kfold_new'
#SBATCH --time=0-22:30:00
#SBATCH --output=bkf_%j.output
#SBATCH --error=bkf_%j.err
#SBATCH --mail-user=r.s.sawhney@student.rug.nl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
pip install pycuda --user
pip install keras-metrics --user
python3 -m venv /data/s4133366/.envs/base_kfold
source /data/s4133366/.envs/base_kfold/bin/activate
pip install pydot
module load Python/3.7.4-GCCcore-8.3.0 CUDA/11.1.1-GCC-10.2.0 cuDNN/7.6.4.38-gcccuda-2019b NCCL/2.4.8-gcccuda-2019b OpenMPI/4.1.1-GCC-10.3.0 TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 Boost.Python/1.71.0-gompi-2019b
python3 kfold_baseline.py

 