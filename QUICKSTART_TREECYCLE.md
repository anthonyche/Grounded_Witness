# 🚀 TreeCycle 快速启动指南

## 立即开始（5 分钟上手）

### 步骤 1: 上传到 HPC

```bash
# 在你的本地机器（Mac）上运行
cd /Users/anthonyche/Desktop/Research
rsync -avz --progress GroundingGEXP/ YOUR_HPC_USERNAME@YOUR_HPC_ADDRESS:/path/to/GroundingGEXP/
```

替换：
- `YOUR_HPC_USERNAME`：你的 HPC 用户名
- `YOUR_HPC_ADDRESS`：HPC 地址（例如 `login.hpc.university.edu`）
- `/path/to/GroundingGEXP/`：HPC 上的目标路径

---

### 步骤 2: SSH 到 HPC

```bash
ssh YOUR_HPC_USERNAME@YOUR_HPC_ADDRESS
```

---

### 步骤 3: 进入项目目录

```bash
cd /path/to/GroundingGEXP
```

---

### 步骤 4: 提交生成任务

```bash
sbatch generate_treecycle.slurm
```

**预期输出**：
```
Submitted batch job 1234567
```

---

### 步骤 5: 监控任务

#### 查看任务状态：
```bash
squeue -u $USER
```

**状态解释**：
- `PD` (Pending): 等待资源
- `R` (Running): 正在运行
- `CG` (Completing): 即将完成
- 如果没有显示：已完成

#### 实时查看输出：
```bash
tail -f logs/treecycle_gen_1234567.out
```
（替换 `1234567` 为你的实际 Job ID）

按 `Ctrl+C` 退出监控。

---

### 步骤 6: 检查结果

#### 等任务完成后（几分钟到几小时），查看完整日志：
```bash
cat logs/treecycle_gen_1234567.out
```

#### 查找成功标志：
```bash
grep "✓ Graph generation completed" logs/treecycle_gen_*.out
```

**成功输出示例**：
```
✓ Graph generation completed successfully!
Elapsed time: 5 minutes 23 seconds
```

#### 查看生成的文件：
```bash
ls -lh datasets/TreeCycle/
```

**应该看到**：
```
-rw-r--r-- 1 user group 2.3G Jan 15 10:30 treecycle_d5_bf15_n813616.pt
```

---

## 📊 理解输出

### 日志文件解读

#### 1. 配置信息
```
TreeCycle Generation Configuration
==================================================
Description: Medium scale (~813K nodes)
  Depth: 5
  Branching factor: 15
  Cycle probability: 0.2
  Node types: 5
==================================================

Expected nodes: 813,616
Estimated memory: 40.7 GB
Estimated time: 10-30 minutes
```

#### 2. 生成过程
```
Tree-Cycle Graph Generator
======================================================================
Parameters:
  Depth: 5
  Branching factor: 15
  Cycle probability: 0.2
  Node types: 5
  Random seed: 42

Building tree structure...
  Root node created: Node 0 (type 2)
  Level 1: 15 nodes (IDs: 1 to 15)
  Level 2: 225 nodes (IDs: 16 to 240)
  Level 3: 3,375 nodes (IDs: 241 to 3,615)
  Level 4: 50,625 nodes (IDs: 3,616 to 54,240)
  Level 5: 759,375 nodes (IDs: 54,241 to 813,615)
✓ Tree built: 813,616 nodes

Adding cycle edges...
  Level 1: Added 2 cycle edges
  Level 2: Added 15 cycle edges
  Level 3: Added 230 cycle edges
  Level 4: Added 5,120 cycle edges
  Level 5: Added 76,238 cycle edges
✓ Cycle edges added: 81,605 total

Graph statistics:
  Nodes: 813,616
  Tree edges: 813,615
  Cycle edges: 81,605
  Total edges: 895,220
  Avg degree: 2.20
  Node types: 5
```

#### 3. 成功标志
```
✓ Generation complete!
======================================================================

Graph saved to: datasets/TreeCycle/treecycle_d5_bf15_n813616.pt
Graph file size: 2.3G
```

---

## ⚠️ 常见问题

### 问题 1: 任务一直在 Pending (PD) 状态

**原因**：
- HPC 资源不足，排队等待
- 请求的资源超出限制

**解决**：
```bash
# 查看队列情况
squeue

# 查看你的任务详细信息
squeue -j 1234567 -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# 如果等待时间过长，修改 Slurm 脚本减少资源请求
# 例如：--mem=32G, --time=03:00:00
```

---

### 问题 2: 任务失败 (没有输出或错误)

**检查错误日志**：
```bash
cat logs/treecycle_gen_1234567.err
```

**常见错误**：

#### A. OOM (Out of Memory)
```
slurmstepd: error: Detected 1 oom_kill event
```

**解决**：
1. 增加内存：修改 `generate_treecycle.slurm` 中的 `#SBATCH --mem=128G`
2. 或减少 cycle_prob：修改脚本中的 `CYCLE_PROB=0.1`

#### B. 模块加载失败
```
module: command not found
```

**解决**：检查 HPC 的环境加载命令，可能需要修改：
```bash
# 替换：
module load Miniconda3
source activate skyexp

# 为：
source /path/to/miniconda3/bin/activate
conda activate skyexp
```

#### C. Python 包缺失
```
ModuleNotFoundError: No module named 'torch_geometric'
```

**解决**：
```bash
# 安装依赖
pip install torch torch_geometric numpy networkx matplotlib
```

---

### 问题 3: 生成文件太小或太大

**预期大小**（粗略估算）：
- 813K 节点：~2-4 GB
- 67M 节点：~100-200 GB
- 1B 节点：~500-1000 GB

**检查**：
```bash
# 查看文件大小
du -h datasets/TreeCycle/treecycle_*.pt

# 如果太小，检查日志是否有警告
grep -i "warning\|error" logs/treecycle_gen_*.out
```

---

## 🎯 下一步：扩大规模

### 成功生成中等规模后，尝试大规模

#### 1. 修改 `generate_treecycle.slurm`

注释掉配置 1，取消注释配置 2：
```bash
# 配置 1: 中等规模（推荐首次运行）
# DEPTH=5
# BRANCHING_FACTOR=15
# CYCLE_PROB=0.2
# NUM_TYPES=5
# DESCRIPTION="Medium scale (~813K nodes)"

# 配置 2: 大规模（如果配置1成功，取消注释这个）
DEPTH=6
BRANCHING_FACTOR=20
CYCLE_PROB=0.15
NUM_TYPES=5
DESCRIPTION="Large scale (~67M nodes)"
```

#### 2. 增加资源

修改 Slurm 参数：
```bash
#SBATCH --time=12:00:00            # 增加到 12 小时
#SBATCH --mem=128G                 # 增加到 128GB
```

#### 3. 重新提交

```bash
sbatch generate_treecycle.slurm
```

---

## 🎓 参数说明

### Depth (深度)
- **含义**：树的层数
- **影响**：指数级影响节点数
- **建议**：3-7

### Branching Factor (分支因子)
- **含义**：每个节点的子节点数
- **影响**：指数级影响节点数
- **建议**：5-30

### Cycle Probability (环概率)
- **含义**：同层节点间添加环边的概率
- **影响**：线性影响边数，但计算时间可能是 O(N²)
- **建议**：
  - 小图（<1M 节点）：0.2-0.3
  - 中图（1M-100M 节点）：0.1-0.2
  - 大图（>100M 节点）：0.01-0.05

### 节点数计算

**公式**：
```
N = (branching_factor^(depth+1) - 1) / (branching_factor - 1)
```

**示例**：
- depth=5, bf=15: N ≈ 813,616
- depth=6, bf=20: N ≈ 67,368,421
- depth=7, bf=30: N ≈ 22,624,137,931 (22.6B)

---

## 📞 需要帮助？

1. **检查日志**：`cat logs/treecycle_gen_*.out`
2. **检查错误**：`cat logs/treecycle_gen_*.err`
3. **查看系统状态**：`squeue -u $USER`
4. **查看磁盘空间**：`quota -s` 或 `df -h`

---

## ✅ 完成标志

当你看到这些，说明成功了：

1. ✓ Slurm 日志中有 "✓ Graph generation completed successfully!"
2. ✓ `datasets/TreeCycle/` 下有 `.pt` 文件
3. ✓ 文件大小合理（几GB到几百GB）
4. ✓ 日志显示正确的节点数和边数

---

## 🚀 完成后

恭喜！你已经成功生成了 TreeCycle 图。

**下一步**：
1. 训练 GNN 模型（使用 `train_treecycle.slurm`，待创建）
2. 运行 witness generation（使用 `benchmark_treecycle.slurm`，待创建）
3. 分析 scalability

详细步骤见 `TREECYCLE_README.md`。
