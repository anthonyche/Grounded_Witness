# Quick Reference: Resume GAT & SAGE Training

## Files Created

✅ `src/Train_Yelp_HPC_GAT_SAGE.py` - Memory-optimized training script
✅ `train_yelp_gat_sage.slurm` - SLURM job script  
✅ `GAT_SAGE_TRAINING_GUIDE.md` - Detailed guide

## What Happened

Your first training run (`train_yelp.slurm`) successfully trained:
- ✓ GCN-1
- ✓ GCN-2  
- ✓ GCN-3

But failed on:
- ✗ GAT (OOM: 26.62 GiB allocation failed)
- ✗ SAGE (probably also OOM)

## Memory Optimizations Applied

| Parameter | Original | Optimized |
|-----------|----------|-----------|
| Hidden dim | 64 | **32** |
| GAT heads | 8 | **2** |
| System RAM | 64GB | **128GB** |
| CUDA alloc | 512MB | **256MB** |
| Time limit | 24h | **48h** |

## Upload & Run (3 Steps)

```bash
# 1. Upload files to HPC
scp src/Train_Yelp_HPC_GAT_SAGE.py username@hpc:GroundingGEXP/src/
scp train_yelp_gat_sage.slurm username@hpc:GroundingGEXP/

# 2. Edit email in SLURM script
ssh username@hpc
cd GroundingGEXP
nano train_yelp_gat_sage.slurm  # Change email and partition

# 3. Submit job
sbatch train_yelp_gat_sage.slurm
```

## Monitor

```bash
# Check status
squeue -u $USER

# Watch logs
tail -f logs/yelp_gat_sage_*.out

# Check errors
tail -f logs/yelp_gat_sage_*.err
```

## Expected Results

**Training time**: ~1-1.5 hours (GPU)

**Output files**:
- `models/Yelp_gat_model.pth`
- `models/Yelp_sage_model.pth`
- `models/Yelp_GAT_SAGE_training_results.json`

## Download Results

```bash
# From local machine
scp username@hpc:GroundingGEXP/models/Yelp_gat_model.pth ./models/
scp username@hpc:GroundingGEXP/models/Yelp_sage_model.pth ./models/
```

## If Still OOM

**Option 1**: Further reduce memory
```python
# Edit src/Train_Yelp_HPC_GAT_SAGE.py line 301:
'hidden_dim': 16,  # Change from 32 to 16

# And line 308:
('gat', 'GAT 3-layer', GAT, {'heads': 1, 'dropout': 0.6}),  # Change heads from 2 to 1
```

**Option 2**: Use CPU (slower but works)
```bash
# Edit train_yelp_gat_sage.slurm:
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=96:00:00
# Remove: #SBATCH --gres=gpu:1
```

## Final Check

After completion, verify all 5 models exist:

```bash
ls -lh models/Yelp_*.pth

# Expected:
# Yelp_gcn1_model.pth   ✓ (from first run)
# Yelp_gcn2_model.pth   ✓ (from first run)
# Yelp_gcn_model.pth    ✓ (from first run)
# Yelp_gat_model.pth    ✓ (new)
# Yelp_sage_model.pth   ✓ (new)
```

## Key Differences from Main Script

| Feature | train_yelp.slurm | train_yelp_gat_sage.slurm |
|---------|------------------|---------------------------|
| Models | All 5 (GCN×3, GAT, SAGE) | Only GAT & SAGE |
| Hidden dim | 64 | 32 |
| GAT heads | 8 | 2 |
| Memory | 64GB | 128GB |
| Time | 24h | 48h |
| Purpose | Initial training | Recovery from OOM |

---

**TL;DR**: Upload 2 files → Edit email → Submit → Wait ~1.5 hours → Download 2 models
