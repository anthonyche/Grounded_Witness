# Quick Start Guide: Training Yelp Models on HPC

## 📋 Pre-requisites

- HPC account with SLURM scheduler
- Python 3.8+ with PyTorch and PyTorch Geometric
- At least 64GB RAM (128GB recommended)
- GPU (optional but recommended)

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Files

Upload these files to your HPC:
```bash
GroundingGEXP/
├── src/
│   ├── Train_Yelp_HPC.py    # Main training script
│   └── utils.py              # Dataset utilities
├── train_yelp.slurm          # GPU training script
├── train_yelp_cpu.slurm      # CPU training script (alternative)
└── config_yelp.yaml          # Configuration (optional)
```

### Step 2: Customize SLURM Script

Edit `train_yelp.slurm`:

```bash
# Line 7: Change partition to your GPU partition name
#SBATCH --partition=gpu

# Line 12: Add your email
#SBATCH --mail-user=your.email@domain.com

# Lines 37-39: Update module loads for your HPC
module load python/3.10
module load cuda/11.8
module load cudnn/8.6.0
```

### Step 3: Submit Job

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch train_yelp.slurm

# Check status
squeue -u $USER

# Monitor output
tail -f logs/yelp_train_*.out
```

## 📊 What Gets Trained

The script trains 5 models automatically:

| Model | Architecture | Layers | Approx. Time (GPU) |
|-------|-------------|--------|-------------------|
| GCN-1 | Graph Convolutional | 1 | 10-15 min |
| GCN-2 | Graph Convolutional | 2 | 15-20 min |
| GCN-3 | Graph Convolutional | 3 | 20-30 min |
| GAT-3 | Graph Attention | 3 | 30-45 min |
| SAGE-3 | GraphSAGE | 3 | 20-30 min |

**Total time: ~2-3 hours on GPU**

## 📁 Output Files

After completion, you'll find:

```
models/
├── Yelp_gcn1_model.pth        # Trained GCN-1 model
├── Yelp_gcn2_model.pth        # Trained GCN-2 model
├── Yelp_gcn_model.pth         # Trained GCN-3 model
├── Yelp_gat_model.pth         # Trained GAT model
├── Yelp_sage_model.pth        # Trained GraphSAGE model
└── Yelp_training_results.json # Training statistics
```

## 🔍 Monitor Progress

```bash
# Check job queue
squeue -u $USER

# View live output
tail -f logs/yelp_train_<JOBID>.out

# View errors
tail -f logs/yelp_train_<JOBID>.err

# Check GPU usage (if on GPU node)
nvidia-smi
```

## 💾 Download Results

After training completes:

```bash
# From your local machine:
scp -r username@hpc:/path/to/GroundingGEXP/models/ ./

# Or just the .pth files:
scp username@hpc:/path/to/GroundingGEXP/models/Yelp_*.pth ./models/
```

## ⚙️ Configuration (Optional)

To customize training parameters, edit `src/Train_Yelp_HPC.py`:

```python
# Find this section in main():
config = {
    'hidden_dim': 64,          # Hidden layer size
    'num_epochs': 200,         # Training epochs
    'learning_rate': 0.01,     # Learning rate
    'weight_decay': 5e-4,      # L2 regularization
    'patience': 50,            # Early stopping patience
}
```

## 🐛 Common Issues

### Issue: Out of memory
**Solution**: Use CPU version
```bash
sbatch train_yelp_cpu.slurm
```

### Issue: Module not found
**Solution**: Install packages on HPC
```bash
pip install --user torch torch-geometric pyyaml
```

### Issue: Job times out
**Solution**: Increase time limit in SLURM script
```bash
#SBATCH --time=48:00:00
```

### Issue: CUDA errors
**Solution**: Check CUDA is available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 📧 Getting Help

1. Check HPC documentation
2. Contact HPC support team
3. Check SLURM manual: `man sbatch`

## ✅ Next Steps

After training, use these models for:
- Node-level explanation with ApxChase
- Baseline comparisons (GNNExplainer, PGExplainer)
- Constraint-based explanations on Yelp dataset

---

**Need more details?** See `HPC_TRAINING_README.md` for comprehensive documentation.
