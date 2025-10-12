#!/bin/bash
#SBATCH --job-name=yelp_train          # Job name
#SBATCH --output=logs/yelp_train_%j.out    # Standard output log
#SBATCH --error=logs/yelp_train_%j.err     # Standard error log
#SBATCH --time=24:00:00                # Time limit (24 hours)
#SBATCH --partition=gpu                # Partition name (adjust for your HPC)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=64G                      # Memory (64GB)
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications
#SBATCH --mail-user=your.email@domain.com  # Your email

# ============================================================================
# SLURM Job Script for Training Yelp GNN Models
# ============================================================================
# This script trains multiple GNN models (GCN-1/2/3, GAT, GraphSAGE) on the
# Yelp dataset for node classification.
#
# Usage:
#   sbatch train_yelp.slurm
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Print system information
echo "System Information:"
echo "  Hostname: $(hostname)"
echo "  CPU cores: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load required modules (adjust for your HPC environment)

# Alternatively, if using conda:
module load Miniconda3
source activate skyexp

echo "Loaded modules:"
module list
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p datasets

# Print Python environment info
echo "Python Environment:"
which python
python --version
echo ""

echo "PyTorch Information:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Change to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Run training script
echo "=========================================="
echo "Starting Yelp GNN Training..."
echo "=========================================="
echo ""

python src/model.py config.yaml output

# Capture exit code
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

# Print disk usage
echo ""
echo "Disk usage in models/:"
du -sh models/
echo ""

# List generated model files
echo "Generated model files:"
ls -lh models/Yelp_*.pth 2>/dev/null || echo "No model files found"
echo ""

exit $EXIT_CODE
