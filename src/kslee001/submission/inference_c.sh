#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=20000MB

for seed in 1000 2000 3000 4000 5000 6000 7000 8000
do
   sbatch inference_c_run.sh --seed $seed
done