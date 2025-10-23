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
每个 DataLoader worker:   ~50-60 GB (复制整个图)
```

### 之前的配置（导致 OOM）
```bash
#SBATCH --mem=128G       # 申请 128GB RAM
NUM_WORKERS=4            # 4个数据加载 worker
```

**实际内存需求**: 
- 主进程: 60GB (数据集)
- Worker 1-4: 4 × 60GB = 240GB
- **总计**: ~300GB > 128GB ❌

### 新的配置（修复后）
```bash
#SBATCH --mem=256G       # 申请 256GB RAM
NUM_WORKERS=0            # 0个 worker（主进程加载）
```

**实际内存需求**:
- 主进程: 60GB (数据集) + 10GB (模型/梯度) = 70GB
- **总计**: ~70GB < 256GB ✅

## 关键修改

### 1. 增加 Slurm RAM 申请
```bash
# 修改前
#SBATCH --mem=128G

# 修改后
#SBATCH --mem=256G
```

### 2. 减少 DataLoader Workers
```bash
# 修改前
NUM_WORKERS=4  # 每个 worker 复制数据集

# 修改后
NUM_WORKERS=0  # 主进程单线程加载（更慢但省内存）
```

## NUM_WORKERS 的权衡

### NUM_WORKERS=0 (当前配置)
- ✅ **内存**: 最低（~70GB）
- ❌ **速度**: 较慢（主进程单线程）
- 适用于: 超大数据集 + RAM 受限

### NUM_WORKERS=1
- ✅ **内存**: 中等（~130GB）
- ✅ **速度**: 中等（1个后台线程）
- 适用于: 如果 256GB RAM 足够

### NUM_WORKERS=4 (之前配置)
- ❌ **内存**: 最高（~300GB）
- ✅ **速度**: 最快（4个并行线程）
- 适用于: 小数据集或超大 RAM (512GB+)

## 性能影响

### 预期训练时间 (NUM_WORKERS=0)
```
每个 epoch: ~20-25 分钟 (比 NUM_WORKERS=4 慢 20-30%)
100 epochs: ~30-40 小时
```

### 如果有更多 RAM 可用
如果集群节点有 512GB+ RAM:
```bash
#SBATCH --mem=512G
NUM_WORKERS=2  # 折中方案
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
