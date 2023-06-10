#!/bin/bash

#SBATCH --job-name=experimental
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --partition=3090
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB

sbatch run_ab.sh --label atel $@
sbatch run_ab.sh --label card $@
sbatch run_ab.sh --label cons $@
sbatch run_ab.sh --label edem $@
sbatch run_ab.sh --label plef $@

