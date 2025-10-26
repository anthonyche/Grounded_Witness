#!/bin/bash
#SBATCH --job-name=treecycle_debug
#SBATCH --output=logs/treecycle_debug_%j.out
#SBATCH --error=logs/treecycle_debug_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G

# Debug version with verbose output

echo "=========================================="
echo "TreeCycle Distributed Benchmark - DEBUG"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Load conda
source ~/.bashrc
conda activate base

# System info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create logs directory
mkdir -p logs

# Run benchmark with debug output
echo "Running benchmark..."
python -u benchmark_treecycle_distributed_v2.py 2>&1 | tee logs/treecycle_debug_${SLURM_JOB_ID}_full.log

echo ""
echo "End time: $(date)"
echo "=========================================="
