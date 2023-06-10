#!/bin/bash
#SBATCH --job-name=dn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=3090
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB

for seed in 9000 10000 11000 12000
do
   sbatch run_dense_ab_121.sh --seed $seed
done