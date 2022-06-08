#!/bin/bash
#SBATCH --job-name='top_pred_imp1'
#SBATCH --time=0-5:00:00
#SBATCH --output=top_pred_imp_1_final_%j.output
#SBATCH --error=top_pred_imp_1_final_%j.err
#SBATCH --mail-user=r.s.sawhney@student.rug.nl
#SBATCH --partition=gpu
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
pip install pycuda --user
pip install keras-metrics --user
python3 -m venv /data/s4133366/.envs/top_1000
source /data/s4133366/.envs/top_1000/bin/activate
pip install tensorflow_addons --user
module load Python/3.7.4-GCCcore-8.3.0 CUDA/11.1.1-GCC-10.2.0 cuDNN/7.6.4.38-gcccuda-2019b NCCL/2.4.8-gcccuda-2019b OpenMPI/4.1.1-GCC-10.3.0 TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 Boost.Python/1.71.0-gompi-2019b
python3 model_predict_tens.py
