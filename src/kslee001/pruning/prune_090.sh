#!/bin/bash
#SBATCH --job-name=pruning
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB

for seed in 1005 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000
do
   sbatch run_prune.sh --target_sparsity 0.90 --seed $seed
done