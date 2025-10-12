# Training GAT and GraphSAGE Models - Recovery Guide

## 问题背景

在训练5个Yelp模型时，GCN-1、GCN-2和GCN-3成功完成，但GAT和GraphSAGE由于GPU内存不足（OOM）失败：

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.62 GiB.
```

## 解决方案

创建了专门的训练脚本和SLURM作业来完成剩余模型的训练，采用以下内存优化策略：

### 内存优化措施

1. **减少隐藏层维度**：从64降到32
2. **减少GAT注意力头数**：从8降到2
3. **增加系统内存**：从64GB增加到128GB
4. **积极的内存清理**：在每个epoch后清理GPU缓存
5. **更小的内存分配块**：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

## 文件说明

### 1. `src/Train_Yelp_HPC_GAT_SAGE.py`

专门训练GAT和GraphSAGE的脚本，包含：

- **内存优化的模型定义**
  - GAT: 2个attention heads（原来是8个）
  - GraphSAGE: 标准配置但使用更小的hidden_dim
  
- **内存管理功能**
  - 自动GPU内存清理
  - 详细的内存使用监控
  - OOM错误捕获和恢复

- **配置参数**
  ```python
  config = {
      'hidden_dim': 32,      # 减少内存使用
      'lr': 0.005,           # 稍低的学习率
      'weight_decay': 5e-4,
      'epochs': 200,
      'patience': 50,
  }
  ```

### 2. `train_yelp_gat_sage.slurm`

SLURM作业脚本，配置：

- **资源请求**
  - GPU: 1个
  - CPU: 8核
  - 内存: **128GB**（增加了一倍）
  - 时间: 48小时

- **环境变量**
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
  export OMP_NUM_THREADS=8
  ```

## 使用步骤

### 步骤1: 上传新文件到HPC

```bash
# 在本地执行
cd ~/Desktop/Research/GroundingGEXP

# 只上传新文件
scp src/Train_Yelp_HPC_GAT_SAGE.py username@hpc:/path/to/GroundingGEXP/src/
scp train_yelp_gat_sage.slurm username@hpc:/path/to/GroundingGEXP/
```

或者上传整个项目（如果已经更新了其他文件）：

```bash
rsync -avz --progress \
  --exclude='datasets/*' \
  --exclude='results/*' \
  --exclude='__pycache__' \
  GroundingGEXP/ username@hpc:/path/to/GroundingGEXP/
```

### 步骤2: 在HPC上修改SLURM脚本

```bash
# SSH登录HPC
ssh username@hpc

# 进入项目目录
cd /path/to/GroundingGEXP

# 编辑SLURM脚本
nano train_yelp_gat_sage.slurm
```

**必须修改的内容：**

```bash
#SBATCH --partition=gpu          # 改为你的GPU分区名
#SBATCH --mail-user=your@email.com   # 改为你的邮箱

# 如果你的HPC模块名不同，修改这里：
module load Miniconda3
source activate skyexp
```

### 步骤3: 提交作业

```bash
# 提交作业
sbatch train_yelp_gat_sage.slurm

# 查看作业状态
squeue -u $USER

# 实时监控日志
tail -f logs/yelp_gat_sage_*.out
```

### 步骤4: 监控训练进度

```bash
# 查看标准输出
tail -f logs/yelp_gat_sage_<JOBID>.out

# 查看错误日志（如果有）
tail -f logs/yelp_gat_sage_<JOBID>.err

# 检查GPU使用
nvidia-smi

# 查看作业详情
scontrol show job <JOBID>
```

## 预期输出

### 训练过程

```
======================================================================
YELP DATASET - GAT & GraphSAGE TRAINING
======================================================================
Device: cuda:0
PyTorch version: 2.0.0
CUDA version: 11.8
GPU: NVIDIA A100-SXM4-40GB
GPU Memory: 40.00 GB
======================================================================

Loading Yelp dataset...
Dataset loaded in 45.23 seconds

Dataset Statistics:
  Nodes: 716,847
  Edges: 13,954,819
  Features: 300
  Classes: 100
  Multi-label: True
  Train nodes: 466,952
  Val nodes: 116,738
  Test nodes: 133,157

######################################################################
# MODEL 1/2: GAT 3-layer
######################################################################

GPU Memory before training:
  Allocated: 0.52 GB
  Reserved: 0.98 GB

Model parameters: 45,300

======================================================================
Training gat on Yelp (Multi-label: True)
======================================================================
Epochs: 200, LR: 0.005, Weight Decay: 0.0005
Device: cuda:0
Patience: 50
======================================================================

Data successfully moved to GPU

Epoch 001 | Loss: 0.6923 | Train: 0.0012/0.4521 | Val: 0.0008/0.4498 | Test: 0.0009/0.4495
  → Saved best model (val_hamming: 0.4498)
Epoch 010 | Loss: 0.2341 | Train: 0.0234/0.7821 | Val: 0.0198/0.7651 | Test: 0.0201/0.7675
  → Saved best model (val_hamming: 0.7651)
...
```

### 成功完成

```
======================================================================
Training completed in 35.67 minutes
Best epoch: 142
Best validation hamming: 0.8534
Final test exact match: 0.0189
Final test hamming: 0.8467
======================================================================

✓ GAT training completed successfully!

GPU Memory after cleanup:
  Allocated: 0.02 GB
  Reserved: 0.50 GB

######################################################################
# MODEL 2/2: GraphSAGE 3-layer
######################################################################

...

======================================================================
TRAINING SUMMARY
======================================================================
Total training time: 4234.56 seconds (70.58 minutes)
Successful models: 2/2
======================================================================

Model           Layers     Params       Val Metric   Test Hamming Time (min)
----------------------------------------------------------------------
GAT             3-layer    45,300       0.8534       0.8467       35.67
SAGE            3-layer    40,164       0.8512       0.8445       34.91
======================================================================

Results saved to: models/Yelp_GAT_SAGE_training_results.json

✓ Successfully trained 2 model(s)!
Models saved in: models/
  - models/Yelp_gat_model.pth
  - models/Yelp_sage_model.pth

======================================================================
Training script completed!
======================================================================
```

## 结果验证

训练完成后，检查模型文件：

```bash
# 在HPC上执行
ls -lh models/Yelp_*.pth

# 应该看到所有5个模型：
# Yelp_gcn1_model.pth   (已有，来自第一次训练)
# Yelp_gcn2_model.pth   (已有，来自第一次训练)
# Yelp_gcn_model.pth    (已有，来自第一次训练)
# Yelp_gat_model.pth    (新训练的)
# Yelp_sage_model.pth   (新训练的)

# 检查训练结果
cat models/Yelp_GAT_SAGE_training_results.json
```

## 下载模型

```bash
# 从本地机器执行
cd ~/Desktop/Research/GroundingGEXP

# 下载新训练的模型
scp username@hpc:/path/to/GroundingGEXP/models/Yelp_gat_model.pth ./models/
scp username@hpc:/path/to/GroundingGEXP/models/Yelp_sage_model.pth ./models/

# 或者下载整个models目录
scp -r username@hpc:/path/to/GroundingGEXP/models/ ./
```

## 预期训练时间

### GPU (NVIDIA A100/V100)
- **GAT**: ~30-40分钟
- **GraphSAGE**: ~30-40分钟
- **总计**: ~1-1.5小时

### 如果仍然OOM

如果即使使用内存优化后仍然OOM，可以尝试：

### 方案A: 进一步减少参数

编辑 `src/Train_Yelp_HPC_GAT_SAGE.py`：

```python
config = {
    'hidden_dim': 16,  # 从32进一步降到16
    ...
}

# 对于GAT
model = model_class(input_dim, hidden_dim, output_dim, heads=1)  # 改为1个头
```

### 方案B: 使用CPU训练

创建CPU版本的SLURM脚本：

```bash
cp train_yelp_gat_sage.slurm train_yelp_gat_sage_cpu.slurm
```

修改：
```bash
#SBATCH --partition=cpu        # 改为CPU分区
#SBATCH --gres=                # 删除GPU请求
#SBATCH --cpus-per-task=16     # 增加CPU核心
#SBATCH --mem=256G             # 增加内存
#SBATCH --time=96:00:00        # 增加时间（CPU更慢）
```

在脚本中注释掉CUDA模块：
```bash
# module load cuda/11.8
# module load cudnn/8.6
```

### 方案C: 使用更大GPU

如果你的HPC有更大内存的GPU（如A100 80GB），可以请求：

```bash
#SBATCH --gres=gpu:a100-80gb:1
# 或
#SBATCH --constraint=gpu_mem:80GB
```

## 故障排查

### 问题1: 仍然OOM

**检查**：
```bash
tail logs/yelp_gat_sage_*.err
```

**解决**：
1. 使用方案A减少hidden_dim到16
2. GAT只使用1个attention head
3. 使用CPU训练（慢但稳定）

### 问题2: 数据集下载失败

**症状**：`ConnectionError` 或 `TimeoutError`

**解决**：
```bash
# 从第一次训练中复制数据集
cp -r datasets/Yelp datasets_backup/
```

### 问题3: 模块加载失败

**检查可用模块**：
```bash
module avail python
module avail cuda
```

**修改SLURM脚本**：
```bash
module load python/3.9  # 或你HPC可用的版本
module load cuda/11.7   # 或你HPC可用的版本
```

## 完成后的检查清单

- [ ] 两个模型都成功训练完成
- [ ] `models/Yelp_gat_model.pth` 存在且大于1MB
- [ ] `models/Yelp_sage_model.pth` 存在且大于1MB
- [ ] `models/Yelp_GAT_SAGE_training_results.json` 存在
- [ ] JSON文件中 `"success": true` 对于两个模型
- [ ] 已下载模型到本地
- [ ] 已备份训练日志

## 下一步

完成GAT和GraphSAGE训练后，你将拥有所有5个Yelp模型：

```
models/
├── Yelp_gcn1_model.pth      ✓
├── Yelp_gcn2_model.pth      ✓
├── Yelp_gcn_model.pth       ✓
├── Yelp_gat_model.pth       ✓ (新)
└── Yelp_sage_model.pth      ✓ (新)
```

接下来可以：

1. **定义Yelp约束**：创建TGD约束用于解释
2. **适配解释算法**：修改ApxChase/HeuChase支持节点级解释
3. **运行解释实验**：对目标节点生成解释
4. **计算指标**：Fidelity-, Conciseness, Coverage

## 快速命令参考

```bash
# 上传文件
scp src/Train_Yelp_HPC_GAT_SAGE.py train_yelp_gat_sage.slurm username@hpc:GroundingGEXP/

# 提交作业
sbatch train_yelp_gat_sage.slurm

# 监控
tail -f logs/yelp_gat_sage_*.out

# 检查结果
ls -lh models/Yelp_{gat,sage}_model.pth

# 下载
scp username@hpc:GroundingGEXP/models/Yelp_{gat,sage}_model.pth models/
```

---

**祝训练顺利！** 🚀

如果遇到任何问题，请查看日志文件或联系HPC支持团队。
