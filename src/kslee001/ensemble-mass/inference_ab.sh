#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition=3090
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=20000MB

for seed in 9000 10000 11000 12000
do
   sbatch inference_ab_run.sh --seed $seed
done