#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition=3090
#SBATCH --cpus-per-task=32
#SBATCH --time=0-12:00:00
#SBATCH --mem=10000MB

source /home/${USER}/.bashrc
source /home/${USER}/anaconda3/bin/activate
conda activate pl

srun python preprocessing.py
