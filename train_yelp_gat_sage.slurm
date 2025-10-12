#!/bin/bash
#SBATCH --job-name=yelp_gat_sage     # Job name
#SBATCH --output=logs/yelp_gat_sage_%j.out    # Standard output log
#SBATCH --error=logs/yelp_gat_sage_%j.err     # Standard error log
#SBATCH --time=48:00:00              # Time limit (48 hours for safety)
#SBATCH --partition=gpu              # Partition name (adjust for your HPC)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # Number of CPU cores
#SBATCH --mem=128G                   # Increased memory to 128GB
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=your.email@domain.com  # Your email

# ============================================================================
# SLURM Job Script for Training GAT & GraphSAGE on Yelp
# ============================================================================
# This script trains GAT and GraphSAGE models that failed in the main run
# due to GPU memory constraints. Uses memory-optimized configurations.
#
# Memory optimizations:
#   - Reduced hidden_dim: 32 (instead of 64)
#   - Fewer GAT attention heads: 2 (instead of 8)
#   - Aggressive memory clearing between operations
#   - Increased system RAM: 128GB
#
# Usage:
#   sbatch train_yelp_gat_sage.slurm
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
module load Miniconda3
source activate skyexp

echo "Loaded modules:"
module list
echo ""

# Set environment variables for memory optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0

# Additional PyTorch memory management
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""

# Print GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Change to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Check if previous models exist
echo "Checking existing models:"
ls -lh models/Yelp_*.pth 2>/dev/null || echo "No previous models found"
echo ""

# Run training script
echo "=========================================="
echo "Starting GAT & GraphSAGE Training..."
echo "=========================================="
echo ""

python src/Train_Yelp_HPC_GAT_SAGE.py

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

# List all generated model files
echo "All model files in models/:"
ls -lh models/Yelp_*.pth 2>/dev/null || echo "No model files found"
echo ""

# Show specifically GAT and SAGE models
echo "GAT and SAGE models:"
ls -lh models/Yelp_gat_model.pth 2>/dev/null || echo "  - GAT model not found"
ls -lh models/Yelp_sage_model.pth 2>/dev/null || echo "  - SAGE model not found"
echo ""

# Print training results if available
if [ -f "models/Yelp_GAT_SAGE_training_results.json" ]; then
    echo "Training results summary:"
    cat models/Yelp_GAT_SAGE_training_results.json
    echo ""
fi

# Print GPU memory usage at the end
if command -v nvidia-smi &> /dev/null; then
    echo "Final GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
    echo ""
fi

exit $EXIT_CODE
