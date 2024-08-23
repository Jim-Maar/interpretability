#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=sae_test
#SBATCH --partition sorcery
#SBATCH --output mats_experiments/slurmout.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:40G
#SBATCH --time=3-0:0:0
#SBATCH --constraint=ARCH:X86

python -u mats_experiments/train_saes.py