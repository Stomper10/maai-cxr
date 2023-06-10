#!/bin/bash

#SBATCH --job-name=experimental
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB

source /home/n1/${USER}/.bashrc
source /home/n1/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./main.py --cluster gsds-ab --batch 64 --epochs 30 $@


