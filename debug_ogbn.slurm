#!/bin/bash
#SBATCH --job-name=ogbn_debug
#SBATCH --output=logs/ogbn_debug_%j.out
#SBATCH --error=logs/ogbn_debug_%j.err
#SBATCH --time=00:30:00              # 30 minutes for debugging
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # Less memory for quick test
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "=================================================="
echo "OGBN-Papers100M Debug Script"
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

# Load modules
module purge
module load Miniconda3
conda activate skyexp

# Print environment
echo "=================================================="
echo "Environment Information:"
echo "=================================================="
which python
python --version
conda list | grep -E "(torch|ogb|geometric)"
echo ""

# Test imports
echo "=================================================="
echo "Testing Python imports:"
echo "=================================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "from ogb.nodeproppred import PygNodePropPredDataset; print('OGB: OK')" 2>&1

echo ""

# Run environment test
echo "=================================================="
echo "Running full environment test:"
echo "=================================================="
python test_ogbn_environment.py

echo ""

# Try to import the training script
echo "=================================================="
echo "Testing training script import:"
echo "=================================================="
python -c "
import sys
sys.path.insert(0, 'src')
print('Attempting to import training modules...')
try:
    import torch
    from torch_geometric.nn import GCNConv
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""

# Test running the script with --help
echo "=================================================="
echo "Testing training script execution:"
echo "=================================================="
python src/Train_OGBN_HPC.py --help 2>&1 | head -30

echo ""
echo "=================================================="
echo "Debug script completed"
echo "End time: $(date)"
echo "=================================================="
