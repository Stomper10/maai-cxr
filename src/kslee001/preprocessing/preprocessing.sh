#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition=3090
#SBATCH --cpus-per-task=32
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G

source /home/n1/${USER}/.bashrc
source /home/n1/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./preprocessing.py $@
