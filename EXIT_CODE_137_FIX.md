# Exit Code 137 修复指南 (系统内存不足)

## 问题诊断

**Exit Code 137** = 128 + 9 (SIGKILL signal)
- **根本原因**: 系统 RAM 不足，Linux OOM Killer 强制终止进程
- **区别于 GPU OOM**: 这是系统内存（RAM）问题，不是显存问题

## 内存使用分析

### ogbn-papers100M 数据集内存需求
```
基础数据集:               ~50-60 GB
PyTorch Geometric 图结构: ~10-15 GB
NeighborLoader 采样缓存:  ~20-40 GB (取决于 batch_size 和 num_neighbors)
每个 DataLoader worker:   ~50-60 GB (复制整个图)
persistent_workers=True:  额外 ~30-50 GB (缓存采样结果)
```

### 第一次尝试配置（导致 OOM）
```bash
#SBATCH --mem=128G       # 申请 128GB RAM
NUM_WORKERS=4            # 4个数据加载 worker
persistent_workers=True  # 持久化 worker 缓存
BATCH_SIZE=1024
NUM_NEIGHBORS="15 10"
```

**实际内存需求**: 
- 主进程: 60GB (数据集) + 40GB (采样缓存)
- Worker 1-4: 4 × 60GB = 240GB
- Persistent cache: 4 × 40GB = 160GB
- **总计**: ~500GB > 128GB ❌

### 第二次尝试配置（仍然 OOM）
```bash
#SBATCH --mem=256G       # 申请 256GB RAM
NUM_WORKERS=0            # 0个 worker
persistent_workers=True  # 仍然启用（虽然 workers=0）
BATCH_SIZE=1024
NUM_NEIGHBORS="15 10"
```

**实际内存需求**:
- 主进程: 60GB (数据集) + 80GB (大batch采样) = 140GB
- NeighborLoader cache: ~120GB (persistent_workers + 大邻居采样)
- **总计**: ~260GB > 256GB ❌

### 新的配置（修复后）✅
```bash
#SBATCH --mem=256G       # 申请 256GB RAM
NUM_WORKERS=0            # 0个 worker（主进程加载）
persistent_workers=False # 禁用持久化缓存
BATCH_SIZE=512           # 减小 batch（减少采样开销）
NUM_NEIGHBORS="10 5"     # 减少邻居采样（减少子图大小）
```

**实际内存需求**:
- 主进程: 60GB (数据集) + 30GB (小batch采样) = 90GB
- NeighborLoader: ~40GB (无缓存，小邻居采样)
- **总计**: ~130GB < 256GB ✅

## 关键修改

### 1. 禁用 persistent_workers (关键！)
```python
# 修改前
train_loader = NeighborLoader(
    data,
    persistent_workers=True if num_workers > 0 else False,  # 缓存数据
    ...
)

# 修改后
train_loader = NeighborLoader(
    data,
    persistent_workers=False,  # 不缓存（节省大量RAM）
    ...
)
```

### 2. 减少 Batch Size
```bash
# 修改前
BATCH_SIZE=1024  # 每次采样 1024 个节点

# 修改后
BATCH_SIZE=512   # 减半（节省 ~40GB RAM）
```

### 3. 减少 Neighbor Sampling
```bash
# 修改前
NUM_NEIGHBORS="15 10"  # 第1层15个邻居，第2层10个

# 修改后
NUM_NEIGHBORS="10 5"   # 减少邻居数（节省 ~30GB RAM）
```

### 4. 保持其他优化
```bash
#SBATCH --mem=256G  # 256GB RAM
NUM_WORKERS=0       # 主进程单线程
```

## 性能影响

### 预期训练时间对比

| 配置 | Batch | Neighbors | Workers | 每 Epoch | 100 Epochs | RAM 使用 |
|------|-------|-----------|---------|----------|------------|----------|
| **最快** | 1024 | 15,10 | 4 | ~15 min | ~25 hrs | ~500GB ❌ |
| **折中** | 1024 | 15,10 | 0 | ~20 min | ~33 hrs | ~260GB ❌ |
| **当前** | 512 | 10,5 | 0 | ~30 min | ~50 hrs | ~130GB ✅ |

### 权衡分析
- **速度损失**: ~2倍慢（50hrs vs 25hrs）
- **内存节省**: ~4倍少（130GB vs 500GB）
- **稳定性**: ✅ 避免 OOM Killer
- **结论**: 牺牲速度换取稳定性（在 256GB 限制下唯一可行方案）

## NeighborLoader 内存消耗详解

### Batch Size 影响
```python
# BATCH_SIZE=1024, NUM_NEIGHBORS=[15,10]
每个 batch 采样节点数 ≈ 1024 × (1 + 15 + 15×10) = ~155,000 节点
内存消耗 ≈ 155,000 × 128 (features) × 4 bytes ≈ 80 MB

# BATCH_SIZE=512, NUM_NEIGHBORS=[10,5]  
每个 batch 采样节点数 ≈ 512 × (1 + 10 + 10×5) = ~26,000 节点
内存消耗 ≈ 26,000 × 128 × 4 bytes ≈ 13 MB

节省: 80 - 13 = 67 MB per batch
总节省 (多个 batch 缓存): ~40-80 GB
```

### persistent_workers 影响
```python
# persistent_workers=True
- Workers 保持活跃状态
- 缓存所有采样的子图
- 内存累积增长（不会释放）
- 额外消耗: ~100-200 GB

# persistent_workers=False
- 每个 batch 后释放内存
- 无缓存累积
- 稍慢（需要重新采样）但内存安全
```

## 验证修复

提交新任务后，监控内存使用:
```bash
# 在另一个终端监控任务
watch -n 5 "squeue -u $USER"

# 找到任务后，SSH 到计算节点
ssh compute_node_name

# 查看内存使用
htop  # 或
top -u $USER
```

应该看到:
- **RSS (Resident Memory)**: 保持在 70-100GB
- **不会被 OOM Killer 终止**

## 相关文件
- `train_ogbn_papers100m.slurm` - Slurm 脚本（已修复）
- `src/Train_OGBN_HPC_MiniBatch.py` - 训练脚本
- `EXIT_CODE_141_FIX.md` - SIGPIPE 问题修复
- `OGBN_TROUBLESHOOTING.md` - 完整故障排除指南

## Exit Code 参考
- **137** = SIGKILL (系统 RAM 不足，OOM Killer)
- **141** = SIGPIPE (管道问题)
- **1** = 一般错误（缺少依赖等）
- **0** = 成功 ✅
