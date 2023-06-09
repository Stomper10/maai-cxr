#!/bin/bash

#SBATCH --job-name=cn-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=50000MB

source /home/${USER}/.bashrc
source /home/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./main_conv_small.py --cluster gsds-c --backbone convnext --epochs 5 --seed 1005 $@


