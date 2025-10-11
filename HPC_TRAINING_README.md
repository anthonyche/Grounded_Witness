# Yelp Dataset Training on HPC

This directory contains scripts for training GNN models on the Yelp dataset on HPC clusters.

## Files

- `src/Train_Yelp_HPC.py`: Main training script (standalone, includes all model definitions)
- `train_yelp.slurm`: SLURM script for GPU training
- `train_yelp_cpu.slurm`: SLURM script for CPU training (if GPU unavailable)

## Models Trained

The script trains the following models:
1. **GCN-1**: 1-layer Graph Convolutional Network
2. **GCN-2**: 2-layer Graph Convolutional Network
3. **GCN-3**: 3-layer Graph Convolutional Network
4. **GAT-3**: 3-layer Graph Attention Network
5. **GraphSAGE-3**: 3-layer GraphSAGE

## Dataset

- **Name**: Yelp
- **Type**: Node classification (Multi-label)
- **Nodes**: ~716,000
- **Edges**: ~14M
- **Features**: 300-dim
- **Labels**: 100 classes (multi-label)
- **Splits**: Pre-defined train/val/test masks

## Usage on HPC

### 1. Prepare your environment

Make sure you have the required packages:
```bash
pip install torch torch-geometric pyyaml numpy
```

### 2. Upload files to HPC

Upload the following to your HPC:
```bash
# From your local machine:
scp -r GroundingGEXP/ username@hpc.server.edu:/path/to/your/directory/
```

### 3. Customize SLURM script

Edit `train_yelp.slurm` or `train_yelp_cpu.slurm`:

```bash
# Update these lines according to your HPC:
#SBATCH --partition=gpu          # Your GPU partition name
#SBATCH --mail-user=your@email.com  # Your email

# Update module loads:
module load python/3.10           # Your Python module
module load cuda/11.8             # Your CUDA version
```

### 4. Submit job

For GPU training:
```bash
cd GroundingGEXP
mkdir -p logs
sbatch train_yelp.slurm
```

For CPU training:
```bash
sbatch train_yelp_cpu.slurm
```

### 5. Monitor job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/yelp_train_<JOBID>.out

# View errors
tail -f logs/yelp_train_<JOBID>.err
```

## Output

### Trained Models

Models will be saved in `models/` directory:
- `models/Yelp_gcn1_model.pth`
- `models/Yelp_gcn2_model.pth`
- `models/Yelp_gcn_model.pth`
- `models/Yelp_gat_model.pth`
- `models/Yelp_sage_model.pth`

### Training Results

Training statistics will be saved in:
- `models/Yelp_training_results.json`

Format:
```json
[
  {
    "model": "gcn1",
    "description": "GCN 1-layer",
    "layers": 1,
    "params": 12345,
    "val_metric": 0.8234,
    "test_acc": 0.8156,
    "train_time": 456.78,
    "path": "models/Yelp_gcn1_model.pth"
  },
  ...
]
```

## Configuration

Default configuration in `Train_Yelp_HPC.py`:
```python
config = {
    'data_root': './datasets',
    'random_seed': 42,
    'hidden_dim': 64,
    'num_epochs': 200,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'patience': 50,  # Early stopping patience
}
```

To modify, edit the `config` dictionary in `main()` function.

## Performance Expectations

### GPU (e.g., NVIDIA V100)
- GCN-1: ~10-15 minutes
- GCN-2: ~15-20 minutes
- GCN-3: ~20-30 minutes
- GAT-3: ~30-45 minutes
- GraphSAGE-3: ~20-30 minutes

**Total: ~2-3 hours**

### CPU (16 cores)
- GCN-1: ~2-3 hours
- GCN-2: ~3-4 hours
- GCN-3: ~4-6 hours
- GAT-3: ~6-10 hours
- GraphSAGE-3: ~4-6 hours

**Total: ~20-30 hours**

## Troubleshooting

### Out of Memory (GPU)
Reduce batch size or use CPU:
```bash
sbatch train_yelp_cpu.slurm
```

### Module not found
Make sure Python environment has required packages:
```bash
# On HPC login node:
pip install --user torch torch-geometric pyyaml
```

### Job killed
Increase memory or time:
```bash
#SBATCH --mem=128G
#SBATCH --time=48:00:00
```

### CUDA errors
Check CUDA compatibility:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Post-Training

After training completes, download models:
```bash
# From your local machine:
scp -r username@hpc.server.edu:/path/to/GroundingGEXP/models/ ./
```

Then use these models for explanation experiments with:
- ApxChase
- HeuChase
- GNNExplainer
- PGExplainer

## Support

For issues specific to your HPC environment, consult:
1. Your HPC documentation
2. HPC support team
3. SLURM manual: `man sbatch`

## Notes

- The script automatically detects and uses GPU if available
- Multi-label classification uses binary cross-entropy loss
- Models use log_softmax output layer
- Early stopping based on validation hamming accuracy
- All models follow the naming convention: `Yelp_{model_name}_model.pth`
