#!/bin/bash
#SBATCH --job-name=generate_cache
#SBATCH --output=logs/generate_cache_%j.out
#SBATCH --error=logs/generate_cache_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo "=================================================="
echo "SLURM Job: Generate Subgraph Cache for OGBN-Papers100M"
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

# Create cache directory
mkdir -p cache/subgraphs
mkdir -p logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Generate cache by running a small benchmark (extracts and caches subgraphs)
echo "Generating cache by extracting subgraphs for 50 nodes..."
echo "This will load the full dataset once and cache L-hop subgraphs."
echo ""

python src/benchmark_ogbn_distributed.py \
    --num_nodes 50 \
    --workers 2 \
    --explainers heuchase

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Cache generation completed successfully!"
    echo ""
    echo "Cache location: cache/subgraphs/"
    ls -lh cache/subgraphs/
    echo ""
    echo "You can now run benchmarks with --use-cache-only flag"
    echo "to avoid loading the full dataset again."
else
    echo "ERROR: Cache generation failed with exit code $EXIT_CODE"
fi
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
