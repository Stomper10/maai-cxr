#!/bin/bash
#SBATCH --job-name=dn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=3090
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB

for seed in 1005 317 1203 7613
do
   sbatch run_dense_ab.sh --seed $seed
done