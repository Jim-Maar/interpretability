#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=othello-gpt-probes
#SBATCH --partition sorcery
#SBATCH --output slurmout.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=3-0:0:0
#SBATCH --constraint=ARCH:X86
python -u training_probes.py