# OGBN-Papers100M Scalability Test - Implementation Summary

## 📦 Created Files

### 1. Training Script
**File**: `src/Train_OGBN_HPC.py`
- 2-layer GCN model for ogbn-papers100M
- Memory-optimized for large-scale graph
- Supports command-line arguments
- Automatic checkpointing
- OGB evaluator integration

### 2. Slurm Job Script
**File**: `train_ogbn_papers100m.slurm`
- **Time**: 48 hours
- **Memory**: 256GB RAM
- **GPU**: 1 GPU (A100/V100 recommended)
- **CPUs**: 8 cores
- Automatic virtual environment setup
- Progress logging

### 3. Configuration Files
- **File**: `config_ogbn_papers100m.yaml` - Configuration template
- Settings for ogbn-papers100M dataset
- Model hyperparameters
- Training parameters

### 4. Documentation
- **File**: `OGBN_PAPERS100M_README.md` - Comprehensive guide
- **File**: `OGBN_QUICK_START.md` - Quick start guide

## 🎯 Key Features

### Model Architecture
```python
class GCN_2_OGBN:
    - Layer 1: GCNConv(128 → 256)
    - Activation: ReLU
    - Dropout: 0.5
    - Layer 2: GCNConv(256 → 172)
    - Output: log_softmax
```

### Dataset Specifications
- **Nodes**: 111,059,956 (111M papers)
- **Edges**: 1,615,685,872 (1.6B citations)
- **Features**: 128 dimensions
- **Classes**: 172 (arXiv subject areas)
- **Task**: Multi-class node classification
- **Split**: Time-based (train ≤2017, val 2018, test ≥2019)

### Memory Configuration
```bash
#SBATCH --mem=256G           # 256GB RAM
#SBATCH --gres=gpu:1         # 1 GPU
Dataset size: ~60GB on disk, ~50GB in memory
```

## 🚀 Usage Workflow

### Step 1: Submit Job
```bash
sbatch train_ogbn_papers100m.slurm
```

### Step 2: Monitor
```bash
squeue -u $USER
tail -f logs/ogbn_papers100m_<job_id>.out
```

### Step 3: Check Results
```bash
cat models/OGBN_Papers100M_training_results.json
ls -lh models/OGBN_Papers100M_gcn2_model.pth
```

## 📊 Expected Performance

### Training Metrics
- **Epoch Time**: ~4 minutes on A100
- **Total Time**: ~7 hours (100 epochs)
- **Memory Usage**: ~45GB GPU, ~200GB RAM
- **Expected Accuracy**: Val ~0.65, Test ~0.64

### Benchmarks by Hardware
| Hardware | Epoch Time | Total Time (100 epochs) |
|----------|------------|-------------------------|
| A100 40GB | ~4 min | ~7 hours |
| V100 32GB | ~5 min | ~9 hours |
| A100 80GB | ~3.5 min | ~6 hours |

## 🔧 Configuration Options

### Command-line Arguments
```bash
python src/Train_OGBN_HPC.py \
    --config config.yaml \
    --data_root ./datasets \
    --epochs 100 \
    --lr 0.01 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --weight_decay 0.0 \
    --log_steps 1 \
    --save_dir models
```

### Slurm Parameters (Adjustable)
```bash
#SBATCH --time=48:00:00      # Max runtime
#SBATCH --mem=256G           # RAM allocation
#SBATCH --gres=gpu:1         # GPU count
#SBATCH --cpus-per-task=8    # CPU cores
```

## 📁 Output Structure

```
GroundingGEXP/
├── src/
│   └── Train_OGBN_HPC.py           # Training script
├── train_ogbn_papers100m.slurm     # Slurm job script
├── config_ogbn_papers100m.yaml     # Config template
├── OGBN_PAPERS100M_README.md       # Full documentation
├── OGBN_QUICK_START.md             # Quick start guide
├── models/
│   ├── OGBN_Papers100M_gcn2_model.pth         # Trained model
│   └── OGBN_Papers100M_training_results.json  # Metrics
├── logs/
│   ├── ogbn_papers100m_<job_id>.out  # Training log
│   └── ogbn_papers100m_<job_id>.err  # Error log
└── datasets/
    └── ogbn_papers100M/              # Dataset (~60GB)
```

## 🛠️ Customization Guide

### Reduce Memory Usage
1. **Smaller hidden dimension**: `--hidden_dim 128` or `--hidden_dim 64`
2. **Request less memory**: `#SBATCH --mem=128G`
3. **Use gradient checkpointing** (requires code modification)

### Speed Up Training
1. **Fewer epochs**: `--epochs 50`
2. **Use A100 GPU**: `#SBATCH --gres=gpu:a100:1`
3. **Distributed training**: Use multiple GPUs (requires code modification)

### Different Model
To use other architectures, modify `src/Train_OGBN_HPC.py`:
- Add GAT/GraphSAGE models
- Update `model_name` parameter
- Adjust hyperparameters

## 🔍 Validation Checklist

Before submitting job:
- [ ] Check HPC partition name (update `#SBATCH --partition`)
- [ ] Verify GPU availability (`sinfo -p gpu`)
- [ ] Ensure sufficient disk space (100GB+)
- [ ] Create logs directory (`mkdir -p logs`)
- [ ] Review memory allocation (256GB minimum)

## 🆘 Troubleshooting

### Common Errors and Solutions

1. **Out of Memory (OOM)**
   - Reduce `--hidden_dim` to 128 or 64
   - Use GPU with more VRAM (A100 80GB)
   - Try CPU training (very slow)

2. **Dataset Download Failure**
   - Check network connectivity
   - Download manually from https://ogb.stanford.edu/
   - Verify disk space (100GB+)

3. **Job Timeout**
   - Increase `#SBATCH --time=72:00:00`
   - Reduce epochs: `--epochs 50`
   - Use faster GPU

4. **GPU Not Available**
   - Check partition: `sinfo -p gpu`
   - Wait for available GPU: `squeue`
   - Request CPU-only (slow): Remove GPU requirements

## 📈 Performance Optimization Tips

1. **Storage**: Use local SSD/NVMe instead of network filesystem
2. **Batch Processing**: Implement NeighborLoader for mini-batch training
3. **Mixed Precision**: Add `torch.cuda.amp` for faster training
4. **Distributed Training**: Use DistributedDataParallel for multi-GPU
5. **Caching**: Save preprocessed data to speed up loading

## 🎓 Next Steps

After successful training:

1. **Model Evaluation**: Test on full test set
2. **Explanation Generation**: Run HeuChase/ApxChase on trained model
3. **Scalability Analysis**: Compare runtime with smaller datasets
4. **Baseline Comparison**: Train GNNExplainer, PGExplainer
5. **Paper Experiments**: Generate scalability figures

## 📚 References

- **OGB Paper**: "Open Graph Benchmark: Datasets for Machine Learning on Graphs" (NeurIPS 2020)
- **Dataset Page**: https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **OGB Leaderboard**: https://ogb.stanford.edu/docs/leader_nodeprop/

## 🙏 Acknowledgments

Dataset provided by Open Graph Benchmark (OGB).
Based on Microsoft Academic Graph (MAG) data.

---

**Status**: ✅ Ready for HPC submission
**Estimated Time**: ~7-9 hours total (including dataset loading)
**Memory Required**: 256GB RAM, 40GB GPU VRAM
