#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=othello_gpt_probes
#SBATCH --partition sorcery
#SBATCH --output slurmout.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:40G
#SBATCH --time=3-0:0:0
#SBATCH --constraint=ARCH:X86

python -u training_probes.py