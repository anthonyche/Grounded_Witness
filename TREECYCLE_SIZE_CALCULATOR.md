# TreeCycle 图大小计算器

## 公式

### 节点数（完全树）
```
N = (b^(d+1) - 1) / (b - 1)
```
其中：
- `b` = branching_factor（分支因子）
- `d` = depth（深度）

**例子**：depth=5, branching_factor=15
```
N = (15^6 - 1) / (15 - 1) = (11,390,625 - 1) / 14 = 813,616 ✓
```

### 边数
- **树边**：N - 1（每个节点除根节点外都有一条到父节点的边）
- **环边**：取决于 `cycle_prob` 和每层节点数
  - 每层最多可能的边对数：`level_nodes * (level_nodes - 1) / 2`
  - 实际添加的边对数：`min(max_pairs * cycle_prob, level_nodes * 10)`
  - 双向边，所以 × 2

### 内存估算
```
Memory (GB) ≈ (2 * num_edges * 8 + num_nodes * num_features * 4) / 1e9
```
- `edge_index`: 2 × num_edges × 8 bytes（int64）
- `x`: num_nodes × num_features × 4 bytes（float32）

---

## 当前配置验证

### 配置二（您目前使用的）
```bash
depth=5, branching_factor=15, cycle_prob=0.2
```

**理论计算**：
- 节点数: 813,616
- 树边数: 813,615
- 环边数: 16,272,042
- 总边数: 17,085,657
- 估算内存: 0.29 GB

**实际结果**：
- 节点数: 813,616 ✅
- 边数: 17,085,657 ✅

**完美匹配！公式正确。**

---

## 推荐配置

### 【小规模配置】（用于快速测试）

#### 小规模-1：~54K 节点（当前的 1/15）
```bash
depth=4
branching_factor=15
cycle_prob=0.2
```
- **节点数**: 54,241
- **边数**: 1,138,782
- **内存**: ~0.02 GB
- **生成时间**: ~5-10 分钟
- **适用场景**: 快速功能验证、本地测试

**生成命令**：
```bash
python TreeCycleGenerator.py --depth 4 --branching-factor 15 --cycle-prob 0.2
```

#### 小规模-2：~111K 节点（当前的 1/7）
```bash
depth=5
branching_factor=10
cycle_prob=0.2
```
- **节点数**: 111,111
- **边数**: 2,333,108
- **内存**: ~0.04 GB
- **生成时间**: ~10-20 分钟
- **适用场景**: 中等功能测试

**生成命令**：
```bash
python TreeCycleGenerator.py --depth 5 --branching-factor 10 --cycle-prob 0.2
```

---

### 【当前配置】（已使用）

#### 中规模：~813K 节点
```bash
depth=5
branching_factor=15
cycle_prob=0.2
```
- **节点数**: 813,616
- **边数**: 17,085,657
- **内存**: ~0.29 GB
- **生成时间**: ~30-60 分钟
- **适用场景**: 标准实验

---

### 【大规模配置】（用于 scalability）

#### 大规模-1：~12M 节点（当前的 16x）
```bash
depth=6
branching_factor=15
cycle_prob=0.15
```
- **节点数**: 12,204,241
- **边数**: 256,288,770
- **内存**: ~4.34 GB
- **生成时间**: ~1-2 小时
- **HPC 配置**: 
  - `--mem=128G`
  - `--time=02:00:00`
  - `--cpus-per-task=20`

**生成命令**：
```bash
python TreeCycleGenerator.py --depth 6 --branching-factor 15 --cycle-prob 0.15
```

**Slurm 脚本配置**：
```bash
# 配置 3: 大规模-1
DEPTH=6
BRANCHING_FACTOR=15
CYCLE_PROB=0.15
NUM_TYPES=5
DESCRIPTION="Large scale (~12M nodes, 256M edges)"
```

#### 大规模-2：~67M 节点（当前的 82x）
```bash
depth=6
branching_factor=20
cycle_prob=0.15
```
- **节点数**: 67,368,421
- **边数**: 1,414,736,476 (~1.4B)
- **内存**: ~23.98 GB
- **生成时间**: ~2-4 小时
- **HPC 配置**: 
  - `--mem=256G`
  - `--time=04:00:00`
  - `--cpus-per-task=20`

**生成命令**：
```bash
python TreeCycleGenerator.py --depth 6 --branching-factor 20 --cycle-prob 0.15
```

**Slurm 脚本配置**：
```bash
# 配置 4: 大规模-2
DEPTH=6
BRANCHING_FACTOR=20
CYCLE_PROB=0.15
NUM_TYPES=5
DESCRIPTION="Very large scale (~67M nodes, 1.4B edges)"
```

**注意**：需要修改 Slurm 头部：
```bash
#SBATCH --mem=256G
#SBATCH --time=04:00:00
```

#### 超大规模：~254M 节点（当前的 312x）
```bash
depth=6
branching_factor=25
cycle_prob=0.1
```
- **节点数**: 254,313,151
- **边数**: 5,340,575,710 (~5.3B)
- **内存**: ~90.54 GB
- **生成时间**: ~4-8 小时
- **HPC 配置**: 
  - `--mem=512G`
  - `--time=08:00:00`
  - `--partition=bigmem` (如果有)

**生成命令**：
```bash
python TreeCycleGenerator.py --depth 6 --branching-factor 25 --cycle-prob 0.1
```

**Slurm 脚本配置**：
```bash
# 配置 5: 超大规模
DEPTH=6
BRANCHING_FACTOR=25
CYCLE_PROB=0.1
NUM_TYPES=5
DESCRIPTION="Ultra large scale (~254M nodes, 5.3B edges)"
```

**注意**：需要修改 Slurm 头部：
```bash
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --partition=bigmem  # 如果 HPC 有这个 partition
```

---

## 配置对比表

| 配置 | Depth | BF | Cycle Prob | 节点数 | 边数 | 内存(GB) | 时间 | 用途 |
|------|-------|----|-----------|---------|---------|---------|---------|--------------------|
| **小规模-1** | 4 | 15 | 0.2 | 54K | 1.1M | 0.02 | ~10min | 快速测试 |
| **小规模-2** | 5 | 10 | 0.2 | 111K | 2.3M | 0.04 | ~20min | 中等测试 |
| **当前** | 5 | 15 | 0.2 | 813K | 17M | 0.29 | ~1h | 标准实验 ✅ |
| **大规模-1** | 6 | 15 | 0.15 | 12M | 256M | 4.34 | ~2h | 大规模实验 |
| **大规模-2** | 6 | 20 | 0.15 | 67M | 1.4B | 23.98 | ~4h | 超大规模实验 |
| **超大规模** | 6 | 25 | 0.1 | 254M | 5.3B | 90.54 | ~8h | 极限测试 |

---

## 使用建议

### 阶段 1：验证（小规模）
使用 **小规模-1** 或 **小规模-2** 快速验证：
- 代码正确性
- Pipeline 完整性
- 资源配置

```bash
# 本地或 HPC
python TreeCycleGenerator.py --depth 4 --branching-factor 15 --cycle-prob 0.2
```

### 阶段 2：标准实验（当前）
使用 **当前配置** 进行正常实验：
- 您已经成功生成 ✅
- 适合大部分实验需求

### 阶段 3：Scalability 测试（大规模）
逐步增加规模：

**步骤 1**：先尝试 **大规模-1** (~12M 节点)
```bash
# 修改 generate_treecycle.slurm
DEPTH=6
BRANCHING_FACTOR=15
CYCLE_PROB=0.15

# 提交任务
sbatch generate_treecycle.slurm
```

**步骤 2**：如果成功，尝试 **大规模-2** (~67M 节点)
```bash
# 修改 generate_treecycle.slurm
DEPTH=6
BRANCHING_FACTOR=20
CYCLE_PROB=0.15

# 修改 Slurm 头部
#SBATCH --mem=256G
#SBATCH --time=04:00:00

# 提交任务
sbatch generate_treecycle.slurm
```

**步骤 3**：如果仍然成功且需要更大规模，尝试 **超大规模** (~254M 节点)
```bash
# 修改 generate_treecycle.slurm
DEPTH=6
BRANCHING_FACTOR=25
CYCLE_PROB=0.1

# 修改 Slurm 头部
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --partition=bigmem  # 如果有

# 提交任务
sbatch generate_treecycle.slurm
```

---

## 为什么降低 cycle_prob？

随着图规模增大，cycle_prob 应该适当降低：

1. **避免过度密集**：
   - 大图的同层节点数巨大
   - cycle_prob=0.2 会导致边数爆炸
   - 例如：depth=6, bf=25 的最后一层有 9,765,625 个节点
     - 最多可能的边对数：~47 trillion
     - cycle_prob=0.2 → ~10 trillion 边 ❌（不可行）
     - cycle_prob=0.1 → ~5 trillion 边（仍需限制）

2. **保持图结构合理**：
   - 环边主要用于引入循环结构
   - 不需要过于密集
   - 稀疏的环边更符合真实世界图

3. **生成和存储效率**：
   - 边数直接影响内存和磁盘空间
   - 边数直接影响 GNN 训练时间
   - 保持边数在合理范围内

---

## 论文中的规模建议

### 主实验（Table/Figure）
- **小规模**：~54K 节点（快速基线）
- **中规模**：~813K 节点（当前，标准对比）
- **大规模**：~12M 节点（scalability 展示）

### Scalability 曲线（单独 Figure）
```
X 轴：图规模（节点数）
Y 轴：运行时间（秒）或内存（GB）

数据点：
- 54K nodes
- 111K nodes
- 813K nodes
- 12M nodes
- 67M nodes (如果资源允许)
```

### 表格示例

| Dataset | Nodes | Edges | Depth | BF | ExhaustChase | HeuChase | PGExplainer |
|---------|-------|-------|-------|----|--------------|-----------|--------------| 
| TreeCycle-S | 54K | 1.1M | 4 | 15 | 12.3s | 8.1s | 15.2s |
| TreeCycle-M | 813K | 17M | 5 | 15 | 45.6s | 28.3s | 32.1s |
| TreeCycle-L | 12M | 256M | 6 | 15 | 180.2s | 95.4s | 102.7s |

---

## 常见问题

### Q1: 为什么不直接生成 depth=7 的图？
**A**: 
- depth=7, bf=15: ~183M 节点
- depth=7, bf=20: ~1.3B 节点
- depth=7, bf=25: ~7.6B 节点

这些规模需要：
- 内存：500GB - 2TB
- 时间：12-48 小时
- 特殊 HPC partition

除非有特殊需求（例如 billion-scale 实验），depth=6 已经足够展示 scalability。

### Q2: 如果 HPC 内存不够怎么办？
**A**: 
1. 降低 `cycle_prob`（减少边数）
2. 使用 **大规模-1** (12M 节点，只需 128GB)
3. 分块生成（修改 TreeCycleGenerator 支持流式生成）

### Q3: 生成时间太长怎么办？
**A**:
1. 使用 `--time=08:00:00` 或更长
2. 添加 checkpoint（定期保存中间结果）
3. 考虑使用更快的存储（例如 SSD）

### Q4: 如何估算自己的配置？
**A**: 使用公式：
```python
def estimate_size(depth, bf, cycle_prob):
    nodes = (bf ** (depth + 1) - 1) // (bf - 1)
    tree_edges = nodes - 1
    # 粗略估算环边（保守估计）
    cycle_edges = nodes * 20 * cycle_prob  # 每个节点平均 20 条环边
    total_edges = tree_edges + cycle_edges
    memory_gb = (2 * total_edges * 8 + nodes * 5 * 4) / 1e9
    return nodes, total_edges, memory_gb
```

---

## 总结

### 立即可用的配置（推荐）✅

**两个更小的配置**：
1. **小规模-1**: `depth=4, bf=15, cycle_prob=0.2` → 54K 节点
2. **小规模-2**: `depth=5, bf=10, cycle_prob=0.2` → 111K 节点

**一个更大的配置**：
3. **大规模-1**: `depth=6, bf=15, cycle_prob=0.15` → 12M 节点

这三个配置已经足够用于：
- 功能验证（小规模）
- 标准实验（当前 813K）
- Scalability 测试（大规模 12M）

如果需要更大规模（67M 或 254M），请确保 HPC 有足够资源。
