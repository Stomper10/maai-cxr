#!/bin/bash

#SBATCH --job-name=tfl-effnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --partition=3090
#SBATCH --mem=180000MB

source /home/n1/${USER}/.bashrc
source /home/n1/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./effnet.py 


