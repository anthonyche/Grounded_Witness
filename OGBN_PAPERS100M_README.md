# OGBN-Papers100M Scalability Test

This directory contains scripts for training a 2-layer GCN on the ogbn-papers100M dataset for scalability testing.

## Dataset Information

**ogbn-papers100M** is a massive citation graph from the Open Graph Benchmark (OGB):
- **111 million nodes** (papers indexed by MAG)
- **1.6 billion edges** (citations)
- **1.5 million labeled nodes** (arXiv papers)
- **172 classes** (arXiv subject areas)
- **Task**: Predict subject areas for arXiv papers
- **Split**: Time-based (train: ≤2017, val: 2018, test: ≥2019)

## Files

- `src/Train_OGBN_HPC.py`: Training script for ogbn-papers100M
- `train_ogbn_papers100m.slurm`: Slurm job script for HPC
- `config_ogbn_papers100m.yaml`: Configuration template

## System Requirements

### Minimum Requirements
- **RAM**: 256GB (dataset is ~60GB, model requires additional memory)
- **GPU**: 40GB+ VRAM (A100 or V100 recommended)
- **Storage**: 100GB free space
- **Time**: ~48 hours for full training

### Alternative Configurations
If resources are limited:
- **CPU training**: Use `--device cpu` (slower but no GPU memory limit)
- **Reduce hidden_dim**: Use 128 or 64 instead of 256
- **Fewer epochs**: Train for 50 epochs instead of 100
- **Mini-batch training**: Implement NeighborLoader (requires code modification)

## Usage

### 1. Update Configuration

Edit `config.yaml` and uncomment the OGBN-Papers100M section:

```yaml
data_name: "ogbn-papers100M"
model_name: "gcn2"
hidden_dim: 256
num_epochs: 100
learning_rate: 0.01
```

Or use the provided template:
```bash
cp config_ogbn_papers100m.yaml config_ogbn.yaml
```

### 2. Submit Slurm Job

```bash
# Submit job to HPC cluster
sbatch train_ogbn_papers100m.slurm
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View output log (replace <job_id> with actual job ID)
tail -f logs/ogbn_papers100m_<job_id>.out

# View error log
tail -f logs/ogbn_papers100m_<job_id>.err
```

### 4. Check Results

After training completes:

```bash
# View training results
cat models/OGBN_Papers100M_training_results.json

# Model checkpoint
ls -lh models/OGBN_Papers100M_gcn2_model.pth
```

## Slurm Configuration Details

The provided Slurm script (`train_ogbn_papers100m.slurm`) requests:
- **Time**: 48 hours
- **Partition**: gpu
- **GPU**: 1 GPU
- **CPUs**: 8 cores
- **RAM**: 256GB
- **Nodes**: 1

You may need to adjust these based on your HPC cluster's configuration:

```bash
#SBATCH --partition=gpu              # Change to your GPU partition name
#SBATCH --gres=gpu:a100:1           # Request specific GPU type
#SBATCH --mem=256G                   # Adjust memory allocation
```

## Training Script Options

The training script supports the following command-line arguments:

```bash
python src/Train_OGBN_HPC.py \
    --config config.yaml \           # Config file path
    --data_root ./datasets \         # Dataset root directory
    --epochs 100 \                   # Number of epochs
    --lr 0.01 \                      # Learning rate
    --hidden_dim 256 \               # Hidden dimension
    --dropout 0.5 \                  # Dropout rate
    --weight_decay 0.0 \             # L2 regularization
    --log_steps 1 \                  # Log frequency
    --save_dir models                # Model save directory
```

## Expected Output

### Training Log
```
==================================================
OGBN-PAPERS100M - GCN TRAINING (2-layer)
==================================================
Device: cuda
PyTorch version: 2.0.1
CUDA version: 11.8
GPU: NVIDIA A100-SXM4-40GB
GPU Memory: 40.00 GB
==================================================

Loading ogbn-papers100M dataset...
Dataset loaded in 15.30 minutes

Dataset Statistics:
  Nodes: 111,059,956
  Edges: 1,615,685,872
  Features: 128
  Classes: 172
  Train nodes: 1,207,179
  Val nodes: 125,265
  Test nodes: 138,949

Model: 2-layer GCN
Parameters: 22,496,172

==================================================
Starting training...
==================================================

Epoch 001 | Loss: 5.1234 | Train: 0.0123 | Val: 0.0145 | Test: 0.0156 | Time: 234.56s
Epoch 002 | Loss: 4.8765 | Train: 0.0234 | Val: 0.0267 | Test: 0.0278 | Time: 231.23s
...
Epoch 100 | Loss: 0.5432 | Train: 0.6789 | Val: 0.6543 | Test: 0.6498 | Time: 228.90s

==================================================
Training completed!
==================================================
Best epoch: 87
Best validation accuracy: 0.6589
Test accuracy: 0.6512
Training time: 382.45 minutes
Model saved to: models/OGBN_Papers100M_gcn2_model.pth
==================================================
```

### Results JSON
```json
{
  "dataset": "ogbn-papers100M",
  "model": "GCN-2",
  "hidden_dim": 256,
  "num_params": 22496172,
  "best_epoch": 87,
  "train_acc": 0.6789,
  "valid_acc": 0.6589,
  "test_acc": 0.6512,
  "training_time": 22947.0,
  "epochs": 100,
  "lr": 0.01,
  "dropout": 0.5,
  "weight_decay": 0.0,
  "timestamp": "2025-10-20 14:35:22"
}
```

## Troubleshooting

### Out of Memory Error
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce hidden dimension: `--hidden_dim 128` or `--hidden_dim 64`
2. Request GPU with more memory: `#SBATCH --gres=gpu:a100:1`
3. Use CPU training (much slower): Remove GPU requirements from Slurm script
4. Implement mini-batch training with `NeighborLoader`

### Dataset Download Failure
```
Error downloading dataset
```

**Solutions:**
1. Download manually: https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M
2. Place in `datasets/ogbn_papers100M/` directory
3. Check network connectivity and firewall settings

### Slow Loading
```
WARNING: This is a large dataset (~60GB). Loading may take several minutes.
```

**This is normal.** The dataset takes 10-20 minutes to load from disk. Consider:
1. Using fast storage (SSD/NVMe) instead of HDD
2. Loading dataset once and saving preprocessed version
3. Using local node storage instead of network filesystem

## Performance Benchmarks

Expected performance on different hardware:

| GPU | RAM | Loading Time | Epoch Time | Total Time (100 epochs) |
|-----|-----|--------------|------------|-------------------------|
| A100 40GB | 256GB | ~15 min | ~4 min | ~7 hours |
| V100 32GB | 256GB | ~18 min | ~5 min | ~9 hours |
| A100 80GB | 512GB | ~12 min | ~3.5 min | ~6 hours |
| CPU (64 cores) | 512GB | ~25 min | ~45 min | ~76 hours |

*Note: Times are approximate and vary based on system configuration*

## Next Steps

After training completes:

1. **Evaluate model**: Load checkpoint and run inference
2. **Run explanations**: Use trained model with HeuChase/ApxChase
3. **Analyze results**: Compare scalability with smaller datasets
4. **Optimize**: Try different architectures (GAT, GraphSAGE)

## References

- OGB Dataset: https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M
- Paper: "Open Graph Benchmark: Datasets for Machine Learning on Graphs" (NeurIPS 2020)
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

## Support

For issues:
1. Check logs in `logs/` directory
2. Verify dataset downloaded correctly
3. Ensure sufficient memory allocation
4. Contact HPC support for cluster-specific issues
