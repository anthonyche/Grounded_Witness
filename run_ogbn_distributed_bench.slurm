#!/bin/bash
#SBATCH --job-name=ogbn_distributed_bench
#SBATCH --output=logs/ogbn_distributed_%j.out
#SBATCH --error=logs/ogbn_distributed_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu              # GPU partition for model inference
#SBATCH --cpus-per-task=20           # 20 CPU cores (enough for 10 workers)
#SBATCH --mem=128G                   # 128GB RAM (reduced from 256G)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # Request 1 GPU for model inference

echo "=================================================="
echo "SLURM Job: Distributed Explainability Benchmark on OGBN-Papers100M"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Load environment
echo "Loading environment..."
module load Miniconda3
source activate skyexp

echo "Python version:"
python --version
echo ""

# Print system info
echo "System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment variables:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo ""

# Check required packages
echo "Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "ERROR: PyTorch not found"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')" || echo "ERROR: PyG not found"
python -c "from ogb.nodeproppred import PygNodePropPredDataset; print('OGB: OK')" || echo "ERROR: OGB not found"
echo ""

# Create results directory
mkdir -p results/ogbn_distributed
mkdir -p logs

# Run benchmark
echo "=================================================="
echo "Starting Distributed Benchmark"
echo "=================================================="
echo ""

# First run: Generate cache by loading dataset and extracting subgraphs
# This loads the full 111M-node graph, extracts subgraphs, and saves to cache/
python src/benchmark_ogbn_distributed.py \
    --num_nodes 100 \
    --workers 20 \
    --explainers apxchase

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
echo ""

# Print result summary
if [ -f "results/ogbn_distributed/complete_results.pkl" ]; then
    echo "Results saved to: results/ogbn_distributed/"
    ls -lh results/ogbn_distributed/
fi

exit $EXIT_CODE
