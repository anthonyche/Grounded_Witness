#!/bin/bash
#SBATCH --job-name=treecycle_gpu_10w
#SBATCH --output=logs/treecycle_gpu_10w_%j.out
#SBATCH --error=logs/treecycle_gpu_10w_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --nodes=1

echo "=================================================="
echo "TreeCycle GPU Test (10 workers, 参考 OGBN 配置)"
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
echo "Device: CUDA"
echo "Workers: 10 (reduced from 20 to match OGBN)"
echo "Memory: 200GB (increased from 128GB to match OGBN)"
echo "Target nodes: 100"
echo "=================================================="

# Run benchmark with 10 workers
echo ""
echo "Starting benchmark with 10 workers..."

# Temporary: modify NUM_WORKERS in code
python -c "
import re
with open('benchmark_treecycle_distributed_v2.py', 'r') as f:
    content = f.read()
# Change NUM_WORKERS = 20 to 10
content = re.sub(r'NUM_WORKERS = 20', 'NUM_WORKERS = 10', content)
# Change DEVICE = 'cpu' to 'cuda'
content = re.sub(r\"DEVICE = 'cpu'\", r\"DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\", content)
with open('benchmark_treecycle_distributed_v2_gpu.py', 'w') as f:
    f.write(content)
print('Created GPU version with 10 workers')
"

python benchmark_treecycle_distributed_v2_gpu.py

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
