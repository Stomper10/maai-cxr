#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=3090
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=20000MB

source /home/n1/${USER}/.bashrc
source /home/n1/${USER}/anaconda3/bin/activate
conda activate tf

srun python ./inference.py --cluster gsds-ab $@
