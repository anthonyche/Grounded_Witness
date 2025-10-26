#!/bin/bash
#SBATCH --job-name=treecycle_cpu_test
#SBATCH --output=logs/treecycle_cpu_test_%j.out
#SBATCH --error=logs/treecycle_cpu_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --nodes=1

echo "=================================================="
echo "TreeCycle CPU Test (避免 CUDA 多进程死锁)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load environment
module load Miniconda3
source activate skyexp

echo "Python version:"
python --version

echo ""
echo "System info:"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo ""
echo "Environment variables:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  PYTHONUNBUFFERED: $PYTHONUNBUFFERED"

# Create output directories
mkdir -p results
mkdir -p logs

echo ""
echo "=================================================="
echo "Configuration"
echo "=================================================="
echo "Device: CPU (避免 CUDA 多进程冲突)"
echo "Workers: 20"
echo "Target nodes: 100"
echo "Explainers: HeuChase only (快速测试)"
echo "=================================================="

# Run benchmark (CPU mode)
echo ""
echo "Starting benchmark..."
python benchmark_treecycle_distributed_v2.py

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully!"
else
    echo "ERROR: Benchmark failed with exit code $EXIT_CODE"
fi
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
