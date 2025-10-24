# TreeCycle Billion-Scale Experiment Guide

## 🎯 目标
在 HPC 上生成 billion-level 的 TreeCycle 图，训练 GNN 模型，并进行大规模 witness generation 测试。

## 📊 实验流程

### 阶段 1: 图生成（当前阶段）

#### 1.1 参数理解

**节点数量公式**：
```
N = (branching_factor^(depth+1) - 1) / (branching_factor - 1)
```

**推荐配置路径**：

| 阶段 | Depth | BF | Cycle Prob | 节点数 | 内存需求 | 时间 | 用途 |
|------|-------|----|-----------:|-------:|---------:|-----:|------|
| 测试 | 3 | 5 | 0.3 | 156 | <1GB | <1分钟 | 本地验证 ✓ |
| 中等 | 5 | 15 | 0.2 | 813K | ~2GB | ~5分钟 | HPC验证 ⭐️ 当前 |
| 大型 | 6 | 20 | 0.15 | 67M | ~20GB | ~1小时 | 性能测试 |
| 超大 | 6 | 25 | 0.1 | 254M | ~60GB | ~3小时 | Scalability |
| Billion | 7 | 30 | 0.05 | 22.6B | ~500GB | ~12小时 | 最终目标 |

#### 1.2 生成图（HPC）

```bash
# 步骤 1: 上传代码到 HPC
rsync -avz --progress GroundingGEXP/ hpc:/path/to/GroundingGEXP/

# 步骤 2: SSH 到 HPC
ssh your_hpc_cluster

# 步骤 3: 进入项目目录
cd /path/to/GroundingGEXP

# 步骤 4: 提交生成任务（从中等规模开始）
sbatch generate_treecycle.slurm
```

**配置选择**：
- 首次运行：使用 `配置 1`（depth=5, bf=15, ~813K nodes）
- 成功后：修改 slurm 脚本，取消注释 `配置 2/3/4`
- Billion-scale：需要修改 Slurm 资源：
  ```bash
  #SBATCH --mem=512G        # 增加到 512GB
  #SBATCH --time=24:00:00   # 增加到 24 小时
  #SBATCH --partition=bigmem # 使用大内存分区（如果有）
  ```

#### 1.3 检查生成结果

```bash
# 查看日志
cat logs/treecycle_gen_JOBID.out

# 查看生成的文件
ls -lh datasets/TreeCycle/

# 应该看到类似文件：
# treecycle_d5_bf15_n813616.pt
# treecycle_d5_bf15_n813616_stats.pkl
```

**成功标志**：
- ✓ 没有 OOM 错误
- ✓ 文件大小合理（几GB到几百GB）
- ✓ 日志显示 "✓ Tree-Cycle graph generated"

---

### 阶段 2: 模型训练（下一步）

#### 2.1 为什么需要训练？
- Witness generation 需要一个训练好的 GNN 模型
- 模型用于评估节点/边的重要性
- 训练可以使用 mini-batch（不需要加载整个图）

#### 2.2 训练脚本（待创建）

```bash
# train_treecycle.slurm
# 使用 NeighborLoader 进行 mini-batch training
# 类似 OGBN-Papers100M 的训练方式
```

**资源需求**：
- GPU: 1 张（用于模型训练）
- 内存: 128-256GB（加载图数据）
- 时间: 2-6 小时

---

### 阶段 3: Witness Generation（最终测试）

#### 3.1 分布式 Witness Generation

使用类似 OGBN-Papers100M 的分布式架构：
- Coordinator: 加载图和模型
- Workers: 并行运行 explainer
- 每个 worker 处理一部分目标节点

#### 3.2 运行基准测试

```bash
# benchmark_treecycle.slurm
# 测试 HeuChase, ApxChase, GNNExplainer
# 在不同规模上测量性能
```

---

## 🚀 快速开始（今天要做的）

### Step 1: 生成中等规模图（验证流程）

```bash
# 在 HPC 上
sbatch generate_treecycle.slurm
```

**预期输出**：
- 节点数: ~813,616
- 边数: ~1.6M - 3.2M（取决于 cycle_prob）
- 文件大小: ~2-4 GB
- 生成时间: 5-10 分钟

### Step 2: 检查结果

```bash
# 查看日志
tail -100 logs/treecycle_gen_*.out

# 确认文件生成
ls -lh datasets/TreeCycle/

# 如果成功，应该看到：
# ✓ Tree-Cycle graph generated
# Nodes: 813,616
# Edges: ~2,000,000
```

### Step 3: 如果成功，升级到大规模

修改 `generate_treecycle.slurm`:
```bash
# 取消注释配置 2
DEPTH=6
BRANCHING_FACTOR=20
CYCLE_PROB=0.15
```

修改 Slurm 资源：
```bash
#SBATCH --mem=128G        # 增加内存
#SBATCH --time=12:00:00   # 增加时间
```

再次提交：
```bash
sbatch generate_treecycle.slurm
```

---

## 📝 注意事项

### 1. 内存管理
- **小心 cycle_prob**：对于大图，即使 0.1 也会产生大量边
- Billion-scale 建议：cycle_prob=0.01-0.05
- 监控内存使用：`squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %C %m"`

### 2. 时间估算
- 节点生成：O(N)，通常很快
- 环边生成：O(N * cycle_prob * layer_size)，可能很慢
- Billion-scale：可能需要 6-24 小时

### 3. 存储空间
- 确保有足够的存储空间（至少 500GB for billion-scale）
- 检查 quota：`quota -s`

### 4. 调试策略
- 从小图开始（depth=5）
- 逐步增大（depth=6, 然后 depth=7）
- 每次成功后再增加规模

---

## 🎓 理论背景

### 为什么选择 TreeCycle？

1. **可控的复杂度**：
   - 通过 depth 和 branching_factor 精确控制规模
   - 通过 cycle_prob 控制图的密集度

2. **适合测试约束传播**：
   - 树结构：测试层次化传播（parent-child）
   - 环结构：测试循环传播（transitivity）
   - 混合结构：测试复杂推理

3. **Scalability**：
   - 可以从 100 节点扩展到 10 billion 节点
   - 结构保持一致，便于比较

### 约束设计

5 个 TGD 约束：
1. **parent_child**: 父子类型传播
2. **transitivity**: 环传递性（A→B→C ⇒ A→C）
3. **hub**: 特定类型枢纽（type 0）
4. **cross_layer**: 跨层桥接（祖父→孙子）
5. **sibling**: 兄弟关系（同父）

---

## 📊 预期结果

### Scalability 指标

我们期望测量：
1. **生成时间** vs 节点数
2. **训练时间** vs 节点数
3. **Witness generation 时间** vs 节点数
4. **内存使用** vs 节点数

### 成功标准

- ✓ 成功生成 billion-level 图
- ✓ 训练出收敛的 GNN 模型
- ✓ 至少一个 explainer 能在合理时间内完成
- ✓ 展示近线性的 scalability

---

## 🔧 故障排除

### 问题 1: OOM (Out of Memory)
**症状**：`slurmstepd: error: Detected 1 oom_kill event`

**解决方案**：
1. 减少 cycle_prob（例如从 0.2 → 0.1）
2. 增加内存分配（`#SBATCH --mem=256G`）
3. 使用分块生成（修改代码，先生成树，再分批添加环）

### 问题 2: 生成时间过长
**症状**：运行超过预期时间

**解决方案**：
1. 减少 depth 或 branching_factor
2. 减少 cycle_prob
3. 检查是否卡在环边生成阶段

### 问题 3: 文件太大无法保存
**症状**：`No space left on device`

**解决方案**：
1. 清理旧文件：`rm -rf datasets/TreeCycle/old_*`
2. 使用压缩存储（修改代码使用 `torch.save(..., pickle_protocol=4)`）
3. 请求更多存储空间

---

## 📚 相关文件

- `TreeCycleGenerator.py`: 图生成器
- `TREECYCLE_PARAMETERS.py`: 参数计算和说明
- `generate_treecycle.slurm`: HPC 生成脚本
- `src/constraints.py`: TreeCycle 约束定义
- `test_treecycle_local.py`: 本地测试脚本

---

## ✅ Checklist

- [x] 本地验证通过（depth=3, bf=5, 156 nodes）
- [ ] HPC 中等规模（depth=5, bf=15, ~813K nodes）⭐️ **当前任务**
- [ ] HPC 大规模（depth=6, bf=20, ~67M nodes）
- [ ] HPC 超大规模（depth=6, bf=25, ~254M nodes）
- [ ] HPC Billion-scale（depth=7, bf=30, ~22.6B nodes）

---

## 🎉 最终目标

成功在 billion-scale 图上：
1. 生成图（~22.6B 节点）
2. 训练 GNN 模型
3. 运行 witness generation
4. 发表 scalability 论文！

Good luck! 🚀
