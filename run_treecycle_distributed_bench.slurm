#!/bin/bash
#SBATCH --job-name=treecycle_bench
#SBATCH --output=logs/treecycle_bench_%j.out
#SBATCH --error=logs/treecycle_bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G

module load Miniconda3
source activate skyexp

python benchmark_treecycle_distributed.py
