# OGBN-Papers100M Quick Start Guide

## 📋 Quick Setup Checklist

- [ ] Review system requirements (256GB RAM, 40GB GPU)
- [ ] Update config.yaml with OGBN settings
- [ ] Create logs directory
- [ ] Submit Slurm job
- [ ] Monitor training progress

## 🚀 Fast Track (3 Steps)

### Step 1: Create Logs Directory
```bash
mkdir -p logs
```

### Step 2: Submit Training Job
```bash
sbatch train_ogbn_papers100m.slurm
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch training log (replace <job_id>)
tail -f logs/ogbn_papers100m_<job_id>.out
```

## 📊 Expected Timeline

- **Dataset Download**: First run only, ~20-30 minutes
- **Dataset Loading**: ~15 minutes per run
- **Training (100 epochs)**: ~6-8 hours on A100
- **Total**: ~7-9 hours

## 🔧 Quick Modifications

### Reduce Training Time
Edit `train_ogbn_papers100m.slurm`:
```bash
EPOCHS=50              # Half the epochs
HIDDEN_DIM=128         # Smaller model
```

### Use Specific GPU
Edit Slurm script:
```bash
#SBATCH --gres=gpu:a100:1    # Request A100
```

### Change Memory Allocation
```bash
#SBATCH --mem=512G           # More memory
```

## 📈 Key Metrics to Watch

Training progress indicators:
- **Epoch 1**: Train ~0.01, Val ~0.01 (random baseline)
- **Epoch 50**: Train ~0.50, Val ~0.48 (learning)
- **Epoch 100**: Train ~0.68, Val ~0.65 (converged)

## 🎯 Success Criteria

✅ Training successful if:
- Dataset loads without errors
- Validation accuracy > 0.60
- Model saved to `models/OGBN_Papers100M_gcn2_model.pth`
- Results JSON created

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `--hidden_dim` to 128 or 64 |
| Slow loading | Use SSD/NVMe storage |
| Job timeout | Increase `#SBATCH --time` |
| GPU unavailable | Wait or use CPU (slow) |

## 📁 Output Files

After completion:
```
models/
  ├── OGBN_Papers100M_gcn2_model.pth          # Model checkpoint
  └── OGBN_Papers100M_training_results.json   # Training metrics
logs/
  ├── ogbn_papers100m_<job_id>.out            # Training log
  └── ogbn_papers100m_<job_id>.err            # Error log
datasets/
  └── ogbn_papers100M/                        # Downloaded dataset (~60GB)
```

## 🔍 Verify Results

```bash
# Check model size
ls -lh models/OGBN_Papers100M_gcn2_model.pth

# View results
cat models/OGBN_Papers100M_training_results.json

# Check dataset size
du -sh datasets/ogbn_papers100M/
```

## 📚 Full Documentation

See `OGBN_PAPERS100M_README.md` for:
- Detailed configuration options
- Troubleshooting guide
- Performance benchmarks
- Advanced usage

## 🎓 Next Steps After Training

1. Load model and run inference
2. Generate explanations with HeuChase/ApxChase
3. Analyze scalability metrics
4. Compare with smaller datasets (Cora, BAShape)
