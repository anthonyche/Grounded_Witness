#!/bin/bash
#SBATCH --job-name=treecycle_bench
#SBATCH --output=logs/treecycle_bench_%j.out
#SBATCH --error=logs/treecycle_bench_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --nodes=1

echo "=================================================="
echo "TreeCycle Distributed Benchmark"
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
echo "  GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for CUDA and multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

echo ""
echo "Environment variables:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"

# Check GPU
echo ""
echo "GPU Check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create output directories
mkdir -p results
mkdir -p logs

echo ""
echo "=================================================="
echo "Configuration"
echo "=================================================="
echo "Dataset: TreeCycle (d5_bf15_n813616)"
echo "Model: TreeCycle_gcn_d5_bf15_n813616.pth"
echo "Workers: 20"
echo "Target nodes: 100"
echo "Hops: 2"
echo "Explainers: HeuChase, ApxChase, ExhaustChase, GNNExplainer, PGExplainer"
echo "Timeout per task: 30 minutes (1800s)"
echo "=================================================="

# Run benchmark
echo ""
echo "Starting benchmark..."
python benchmark_treecycle_distributed_v2.py

echo ""
echo "=================================================="
echo "Benchmark completed"
echo "End time: $(date)"
echo "=================================================="
