#!/bin/bash

#SBATCH --job-name=mass-expert
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB

source /home/${USER}/.bashrc
source /home/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./main.py --cluster gsds-c --backbone densenet --epochs 35 --add_expert $@


