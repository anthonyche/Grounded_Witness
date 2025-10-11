#!/bin/bash
#SBATCH --job-name=yelp_train_cpu      # Job name
#SBATCH --output=logs/yelp_train_cpu_%j.out
#SBATCH --error=logs/yelp_train_cpu_%j.err
#SBATCH --time=48:00:00                # Time limit (48 hours for CPU)
#SBATCH --partition=cpu                # CPU partition
#SBATCH --cpus-per-task=16             # Number of CPU cores
#SBATCH --mem=128G                     # Memory (128GB)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# ============================================================================
# SLURM Job Script for Training Yelp GNN Models (CPU Version)
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load required modules
module purge
module load python/3.10

echo "Loaded modules:"
module list
echo ""

# Set environment variables for CPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p datasets

# Change to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Run training script
echo "=========================================="
echo "Starting Yelp GNN Training (CPU)..."
echo "=========================================="
echo ""

python src/Train_Yelp_HPC.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "Job finished at: $(date)"
echo "=========================================="

exit $EXIT_CODE
