# OGBN Training Failed - Troubleshooting Guide

## 🔍 Common Exit Codes

### Exit Code 137: System Memory (RAM) OOM ⭐ NEW
**Cause**: Linux OOM Killer terminated process due to insufficient RAM

**Symptoms**:
- Job runs for a few seconds/minutes then dies
- No Python error message
- Last output: "Using 'NeighborSampler' without a 'pyg-lib'..."
- System logs show OOM Killer activity

**Fix**: See `EXIT_CODE_137_FIX.md`
```bash
# Quick fix in train_ogbn_papers100m.slurm
#SBATCH --mem=256G        # Increase RAM (was 128G)
NUM_WORKERS=0             # Disable DataLoader workers (was 4)
```

**Why**: Each DataLoader worker copies the entire graph (~60GB). With 4 workers:
- Main process: 60GB
- 4 workers: 4 × 60GB = 240GB
- Total: 300GB > 128GB ❌

### Exit Code 141: SIGPIPE Error
**Cause**: Broken pipe from `yes y |` command

**Fix**: See `EXIT_CODE_141_FIX.md`
```bash
# Remove pipe in Slurm script
python src/Train_OGBN_HPC_MiniBatch.py  # Direct execution
```

### Exit Code 1: General Error
**Cause**: Missing dependencies, import errors, or configuration issues

**Fix**: See sections below

## 🎯 Most Likely Causes (Exit Code 1)

### 1. Missing OGB Library ⭐ (MOST LIKELY)
The `ogb` package is not installed in your conda environment.

**Fix:**
```bash
# On HPC login node
module load Miniconda3
conda activate skyexp
pip install ogb
```

### 2. Import Error
Python can't import required modules (torch, torch_geometric, ogb)

### 3. Dataset Download Failed
OGB dataset download was interrupted or failed silently

## 🛠️ Step-by-Step Diagnosis

### Step 1: Run Debug Script
```bash
# Submit debug job
sbatch debug_ogbn.slurm

# Monitor output
tail -f logs/ogbn_debug_*.out
```

This will check:
- Python environment
- Package versions
- Import capabilities
- Script execution

### Step 2: Check Error Log
```bash
# Find latest error log
ls -lt logs/ogbn_papers100m_*.err | head -1

# View full error
cat logs/ogbn_papers100m_<job_id>.err
```

Look for:
- `ModuleNotFoundError: No module named 'ogb'`
- `ImportError`
- `RuntimeError`
- Any Python traceback

### Step 3: Manual Environment Test
```bash
# On HPC login node
module load Miniconda3
conda activate skyexp

# Run environment test
python test_ogbn_environment.py
```

This will identify missing packages.

### Step 4: Check Package Installation
```bash
# List installed packages
conda list | grep -E "(torch|ogb|geometric)"

# Should show:
#   pytorch                   2.x.x
#   torch-geometric           2.x.x
#   ogb                       1.3.x (or similar)
```

## ✅ Quick Fixes

### Fix 1: Install OGB
```bash
module load Miniconda3
conda activate skyexp
pip install ogb
```

### Fix 2: Reinstall PyTorch Geometric (if needed)
```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Fix 3: Test Imports Manually
```bash
python -c "from ogb.nodeproppred import PygNodePropPredDataset; print('OGB OK')"
```

Should print "OGB OK" without errors.

## 📋 Checklist Before Resubmitting

- [ ] OGB installed: `pip list | grep ogb`
- [ ] Environment test passes: `python test_ogbn_environment.py`
- [ ] Script runs with --help: `python src/Train_OGBN_HPC.py --help`
- [ ] datasets/ directory exists and writable
- [ ] logs/ directory exists

## 🚀 After Fixing

### Re-submit Training Job
```bash
# Clean old logs (optional)
rm logs/ogbn_papers100m_*.out logs/ogbn_papers100m_*.err

# Submit again
sbatch train_ogbn_papers100m.slurm

# Monitor
squeue -u $USER
tail -f logs/ogbn_papers100m_*.out
```

## 📊 Expected Behavior After Fix

You should see:
```
==================================================
OGBN-PAPERS100M - GCN TRAINING (2-layer)
==================================================
Device: cuda
PyTorch version: 2.x.x
CUDA version: 11.8 (or 12.x)
GPU: NVIDIA L40S (or your GPU)
==================================================

Loading ogbn-papers100M dataset...
WARNING: This is a large dataset (~60GB). Loading may take several minutes.
```

Then dataset download starts (first run only).

## 🔧 Advanced Debugging

### Check Python Executable
```bash
which python  # Should be in conda environment
```

### Check PYTHONPATH
```bash
echo $PYTHONPATH
```

### Verbose Python Execution
```bash
python -v src/Train_OGBN_HPC.py --help 2>&1 | grep -i "import ogb"
```

### Check Disk Space
```bash
df -h .  # Should have 100GB+ free
```

## 📞 Get Help

If issue persists, provide:
1. Output of `debug_ogbn.slurm`
2. Content of error log: `logs/ogbn_papers100m_*.err`
3. Output of `python test_ogbn_environment.py`
4. Output of `conda list | grep -E "(torch|ogb)"`

## 🎓 Prevention

Add to your job scripts:
```bash
# Always check environment before training
python test_ogbn_environment.py || exit 1
```

This prevents wasting GPU hours on broken environments.
