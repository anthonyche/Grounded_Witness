# ogbn-papers100M 内存优化指南

## 🎯 快速诊断

**看到 Exit Code 137？** → 系统 RAM 不足（OOM Killer）

## ✅ 当前最优配置（256GB RAM 限制下）

### Slurm 资源
```bash
#SBATCH --mem=256G       # 256GB RAM
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
```

### 训练参数
```bash
BATCH_SIZE=512           # 小 batch（节省采样内存）
NUM_NEIGHBORS="10 5"     # 小邻居采样（减少子图大小）
NUM_WORKERS=0            # 主进程加载（避免多进程复制）
```

### Python 代码
```python
# 所有 NeighborLoader 都要设置
train_loader = NeighborLoader(
    data,
    batch_size=512,
    num_neighbors=[10, 5],
    num_workers=0,
    persistent_workers=False,  # 关键！禁用缓存
    ...
)
```

## 📊 内存消耗分解

| 组件 | 内存消耗 | 说明 |
|------|----------|------|
| **数据集加载** | ~60 GB | ogbn-papers100M 基础数据 |
| **图结构** | ~15 GB | PyG 的 COO 格式边索引 |
| **NeighborLoader 采样** | ~30 GB | 子图采样和缓存 |
| **模型 + 梯度** | ~5 GB | GCN 参数和反向传播 |
| **其他开销** | ~20 GB | Python 运行时、系统缓存 |
| **总计** | **~130 GB** | ✅ 在 256GB 限制内 |

## 🔧 优化参数对比

### Batch Size
| 值 | 内存影响 | 速度影响 | 推荐 |
|----|----------|----------|------|
| 2048 | 非常高 (~200 GB) | 最快 | ❌ OOM |
| 1024 | 高 (~140 GB) | 快 | ❌ OOM |
| 512 | 中 (~90 GB) | 中 | ✅ 当前 |
| 256 | 低 (~60 GB) | 慢 | 备选 |

### Neighbor Sampling
| 值 | 采样节点数 | 内存影响 | 推荐 |
|----|-----------|----------|------|
| [20, 15] | ~154k/batch | 很高 | ❌ |
| [15, 10] | ~77k/batch | 高 | ❌ |
| [10, 5] | ~26k/batch | 中 | ✅ 当前 |
| [5, 3] | ~8k/batch | 低 | 备选 |

### NUM_WORKERS
| 值 | 内存消耗 | 说明 |
|----|----------|------|
| 4 | +240 GB | 每个 worker 复制数据集 ❌ |
| 2 | +120 GB | 仍然太大 ❌ |
| 1 | +60 GB | 可能触发 OOM ⚠️ |
| 0 | +0 GB | 无额外复制 ✅ 当前 |

### persistent_workers
| 值 | 内存影响 | 说明 |
|----|----------|------|
| True | +50-200 GB | 缓存所有采样结果，累积增长 ❌ |
| False | +0 GB | 每个 batch 后释放 ✅ 当前 |

## 🚀 如果仍然 OOM 的应急方案

### 方案 1: 进一步减小 batch（推荐）
```bash
BATCH_SIZE=256
NUM_NEIGHBORS="8 4"
```
- 内存: ~80 GB
- 速度: 慢 2-3 倍

### 方案 2: 申请更多 RAM
```bash
#SBATCH --mem=512G  # 如果集群有更大节点
BATCH_SIZE=1024
NUM_NEIGHBORS="15 10"
NUM_WORKERS=1  # 可以启用 1 个 worker
```

### 方案 3: CPU-only 训练
```bash
# 在 Slurm 脚本中
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
# 不申请 GPU

# 在训练脚本中
device = 'cpu'
```
- 内存: 可以用 swap（慢但稳定）
- 速度: 慢 10-50 倍

## 📈 性能预期

### 当前配置 (BATCH=512, NEIGHBORS=[10,5], WORKERS=0)
```
数据集加载: ~5-10 分钟
每个 epoch: ~30-40 分钟
100 epochs: ~50-70 小时（2-3 天）
```

### 监控命令
```bash
# 提交任务
sbatch train_ogbn_papers100m.slurm

# 监控内存使用
watch -n 5 "squeue -u $USER"
# SSH 到计算节点后
htop  # 看 RES 列（常驻内存）应保持在 130-150 GB

# 监控训练进度
tail -f logs/ogbn_papers100m_*.out
```

## ⚠️ 常见错误

### 错误 1: 忘记更新所有 NeighborLoader
```python
# ❌ 错误：只改了 train_loader
train_loader = NeighborLoader(..., persistent_workers=False)
val_loader = NeighborLoader(..., persistent_workers=True)  # 仍会 OOM！

# ✅ 正确：三个都改
train_loader = NeighborLoader(..., persistent_workers=False)
val_loader = NeighborLoader(..., persistent_workers=False)
test_loader = NeighborLoader(..., persistent_workers=False)
```

### 错误 2: NUM_WORKERS > 0 但忘记调整 RAM
```bash
# ❌ 错误
#SBATCH --mem=256G
NUM_WORKERS=2  # 需要 +120GB = 总共 ~250GB，接近极限

# ✅ 正确
#SBATCH --mem=512G  # 或者
NUM_WORKERS=0      # 保持 256G
```

### 错误 3: Batch size 在 Slurm 和 Python 不一致
```bash
# Slurm 脚本
BATCH_SIZE=512

# Python 脚本（命令行参数）
python Train_OGBN_HPC_MiniBatch.py --batch_size 1024  # ❌ 被覆盖！

# 确保传递正确参数
python Train_OGBN_HPC_MiniBatch.py --batch_size $BATCH_SIZE  # ✅
```

## 🎓 理解 NeighborLoader 内存机制

### 为什么 batch_size=1024 需要这么多内存？

```
假设 batch_size=1024, num_neighbors=[15, 10]

第 0 层（目标节点）: 1024 个节点
第 1 层（1-hop 邻居）: 1024 × 15 = 15,360 个节点
第 2 层（2-hop 邻居）: 15,360 × 10 = 153,600 个节点

总采样节点数: 1024 + 15,360 + 153,600 ≈ 170,000 个节点

每个节点:
- Features: 128 × 4 bytes = 512 bytes
- Labels: 4 bytes
- 其他元数据: ~100 bytes

每个 batch 内存: 170,000 × 616 bytes ≈ 105 MB

如果 NeighborLoader 缓存 1000 个 batch:
1000 × 105 MB = 105 GB！（这就是为什么会 OOM）
```

### persistent_workers=False 如何节省内存？

```python
# persistent_workers=True
for batch in loader:
    process(batch)
    # batch 处理完后，NeighborLoader 仍然保存在内存中
    # 累积 N 个 batch = N × 105 MB

# persistent_workers=False
for batch in loader:
    process(batch)
    # batch 处理完后，内存立即释放
    # 始终只占用 1 × 105 MB
```

## 📚 相关文档
- `EXIT_CODE_137_FIX.md` - Exit Code 137 详细分析
- `OGBN_TROUBLESHOOTING.md` - 完整故障排除
- `train_ogbn_papers100m.slurm` - 优化后的 Slurm 脚本
- `src/Train_OGBN_HPC_MiniBatch.py` - 优化后的训练脚本
