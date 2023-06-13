#!/bin/bash
#SBATCH --job-name=run_mass
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB

for seed in 1000 2000 3000 4000 5000 6000 7000 8000
do
   sbatch run_dense_c_121.sh --seed $seed
done
