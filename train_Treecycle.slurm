#!/bin/bash
#SBATCH --job-name=train_treecycle_gcn
#SBATCH --output=logs/train_treecycle_gcn_%j.out
#SBATCH --error=logs/train_treecycle_gcn_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load Miniconda3
source activate skyexp

python train_Treecycle.py
