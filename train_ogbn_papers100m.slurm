#!/bin/bash
#SBATCH --job-name=ogbn_papers100m_gcn
#SBATCH --output=logs/ogbn_papers100m_%j.out
#SBATCH --error=logs/ogbn_papers100m_%j.err
#SBATCH --time=12:00:00              # 48 hours (large dataset)
#SBATCH --partition=gpu              # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # 8 CPU cores
#SBATCH --mem=128G                   # 256GB RAM (dataset ~60GB + model + overhead)
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Note: ogbn-papers100M is a very large dataset
# Dataset size: ~60GB on disk, ~50GB in memory
# Recommended: GPU with 40GB+ VRAM (A100/V100) or use CPU with large RAM

echo "=================================================="
echo "SLURM Job: Training GCN on ogbn-papers100M"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Print system info
echo "System Information:"
nvidia-smi
echo ""
echo "CPU Info:"
lscpu | grep "Model name"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Load modules (adjust based on your HPC environment)
module purge
module load Miniconda3
conda activate skyexp

# Check Python environment
echo "Python environment check:"
which python
python --version
echo ""

# Check required packages
echo "Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "ERROR: PyTorch not found"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')" || echo "ERROR: PyG not found"
python -c "from ogb.nodeproppred import PygNodePropPredDataset; print('OGB: OK')" || echo "ERROR: OGB not found - installing..."

# Install OGB if missing
python -c "from ogb.nodeproppred import PygNodePropPredDataset" 2>/dev/null || pip install ogb

echo "Package check complete."
echo ""

# Run environment test before training
echo "=================================================="
echo "Running environment test..."
echo "=================================================="
python test_ogbn_environment.py
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Environment test failed!"
    echo "Please install missing packages and try again."
    exit 1
fi

echo ""
echo "Environment test passed! Proceeding with training..."
echo ""

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment variables:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Training configuration
EPOCHS=100
LR=0.01
HIDDEN_DIM=256
DROPOUT=0.5
WEIGHT_DECAY=0.0
LOG_STEPS=1

echo "=================================================="
echo "Training Configuration:"
echo "=================================================="
echo "  Dataset: ogbn-papers100M"
echo "  Model: 2-layer GCN"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Dropout: $DROPOUT"
echo "  Weight decay: $WEIGHT_DECAY"
echo "=================================================="
echo ""

# Run training
echo "Starting training..."
echo ""

# Add error handling and verbose output
set -x  # Print commands as they execute
# Direct execution (no srun needed for single-node jobs)
yes y | python src/Train_OGBN_HPC.py \
    --config config.yaml \
    --data_root ./datasets \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --log_steps $LOG_STEPS \
    --save_dir models 2>&1 | tee training_log.txt

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +x

# If training failed, show error details
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=================================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "=================================================="
    echo ""
    echo "Last 50 lines of output:"
    tail -50 training_log.txt
    echo ""
fi

echo ""
echo "=================================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

# Print final GPU memory usage
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU memory usage:"
    nvidia-smi
fi

# Print disk usage
echo ""
echo "Disk usage:"
du -sh datasets/
du -sh models/
echo ""



exit $TRAIN_EXIT_CODE
