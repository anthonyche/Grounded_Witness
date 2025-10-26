#!/bin/bash
#SBATCH --job-name=treecycle_quick
#SBATCH --output=logs/treecycle_quick_%j.out
#SBATCH --error=logs/treecycle_quick_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Quick test: 5 nodes, 2 workers

echo "=========================================="
echo "TreeCycle Quick Test (5 nodes, 2 workers)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

source ~/.bashrc
conda activate base

mkdir -p logs

# Run with modified parameters
python -u -c "
import sys
sys.path.insert(0, '.')

# Read and modify the benchmark script
with open('benchmark_treecycle_distributed_v2.py', 'r') as f:
    code = f.read()

# Replace parameters for quick test
code = code.replace('NUM_TARGETS = 100', 'NUM_TARGETS = 5')
code = code.replace('NUM_WORKERS = 20', 'NUM_WORKERS = 2')
code = code.replace(
    \"EXPLAINERS = ['heuchase', 'apxchase', 'exhaustchase', 'gnnexplainer', 'pgexplainer']\",
    \"EXPLAINERS = ['heuchase']\"
)

# Execute modified code
exec(code)
" 2>&1 | tee logs/treecycle_quick_${SLURM_JOB_ID}.log

echo ""
echo "End time: $(date)"
