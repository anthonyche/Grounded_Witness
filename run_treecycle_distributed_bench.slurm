#!/bin/bash
#SBATCH --job-name=treecycle_gpu
#SBATCH --output=logs/treecycle_bench_%j.out
#SBATCH --error=logs/treecycle_bench_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --nodes=1

echo "=================================================="
echo "TreeCycle Distributed Benchmark (Hybrid CPU/GPU)"
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
echo "  GPUs requested: 4"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set environment variables for CUDA and multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo ""
echo "Environment variables:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  PYTHONUNBUFFERED: $PYTHONUNBUFFERED"

# Check GPU
echo ""
echo "=================================================="
echo "GPU Information"
echo "=================================================="
nvidia-smi
echo ""
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Create output directories
mkdir -p results
mkdir -p logs

echo ""
echo "=================================================="
echo "Configuration"
echo "=================================================="
echo "Dataset: TreeCycle (d5_bf15_n813616)"
echo "Model: TreeCycle_gcn_d5_bf15_n813616.pth"
echo "Workers: 20 (distributed across 4 GPUs)"
echo "  GPU allocation: Round-robin (5 workers per GPU)"
echo "Target nodes: 100"
echo "Hops: 2"
echo "Strategy: Hybrid CPU/GPU"
echo "  - Model inference: GPU"
echo "  - Computation (constraints, Edmonds): CPU"
echo "Explainers: ApxChase (HeuChase disabled due to Edmonds issue)"
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
