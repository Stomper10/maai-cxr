#!/bin/bash

#SBATCH --job-name=experimental
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB

sbatch run_c.sh --label atel $@
sbatch run_c.sh --label card $@
sbatch run_c.sh --label cons $@
sbatch run_c.sh --label edem $@
sbatch run_c.sh --label plef $@

