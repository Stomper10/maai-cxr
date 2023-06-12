#!/bin/bash
#SBATCH --job-name=pruning
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB

for seed in 1005 317 1203 7613
do
   sbatch run_prune_label_ensemble.sh --target_sparsity 0.90 --seed $seed
done