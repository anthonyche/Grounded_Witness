# Distributed Benchmark Integration Summary

## 概述

成功整合了真实的解释算法（HeuChase, ApxChase, GNNExplainer）到分布式基准测试框架中。

## 已完成的工作

### 1. 更新 `src/benchmark_ogbn_distributed.py`

#### 1.1 导入真实解释器
```python
from heuchase import HeuChase
from apxchase import ApxChase
from baselines import run_gnn_explainer_node
```

#### 1.2 更新 `worker_process()` 函数
- **HeuChase 集成**: 使用 Edmonds-based witness generation
  - 参数: `model`, `Sigma=None`, `L=2`, `k=10`, `B=5`, `m=6`, `noise_std=1e-3`
  - 调用: `explainer._run(H=subgraph, root=target_node)`
  - 返回: `Sigma_star` (coverage), `S_k` (witnesses list)

- **ApxChase 集成**: 使用 streaming edge-insertion chase
  - 参数: `model`, `Sigma=None`, `L=2`, `k=10`, `B=5`
  - 调用: `explainer._run(H=subgraph, root=target_node)`
  - 返回: 与 HeuChase 相同

- **GNNExplainer 集成**: 使用 PyG baseline
  - 参数: `epochs=100`, `lr=0.01`
  - 调用: `run_gnn_explainer_node(model, data, target_node, ...)`
  - 返回: `edge_mask`, `pred`, `prob`

#### 1.3 添加 `explainer_config` 参数
- 传递解释器特定配置到 worker processes
- 支持不同解释器的不同参数需求

### 2. Node Classification 模式

#### 2.1 关键适配
- **Sigma = None**: 不使用约束（constraints），适用于 node classification
- **root = target_node**: 指定要解释的目标节点
- **Model calling**: `model(x, edge_index)` 而非 `model(Data)`
- **Verification**: `_default_verify_witness()` 自动检测模型类型

#### 2.2 Subgraph 提取
- Coordinator 使用 `k_hop_subgraph()` 提取 2-hop 邻居
- 子图包含：
  - `x`: 节点特征
  - `edge_index`: 边索引
  - `y`: 节点标签
  - `target_node`: 目标节点在子图中的索引（重标号后）

### 3. 配置参数

#### 3.1 HeuChase 配置
```python
{
    'Sigma': None,        # 无约束
    'L': 2,               # 2-hop subgraph
    'k': 10,              # window size
    'B': 5,               # budget
    'm': 6,               # Edmonds candidates
    'noise_std': 1e-3,    # noise for diversity
}
```

#### 3.2 ApxChase 配置
```python
{
    'Sigma': None,
    'L': 2,
    'k': 10,
    'B': 5,
}
```

#### 3.3 GNNExplainer 配置
```python
{
    'epochs': 100,        # 训练轮数
    'lr': 0.01,           # 学习率
}
```

### 4. 测试文件

#### 4.1 `test_distributed_explainer.py`
- 测试导入和基本功能
- 使用 dummy subgraph (10 nodes, 20 edges)
- 验证每个解释器可以独立运行
- **用途**: 快速验证集成是否正确

#### 4.2 `test_distributed_quick.py`
- 测试完整的 Coordinator-Worker 架构
- 使用真实 OGBN-Papers100M 数据
- 只测试 5 个节点，2 个 workers
- **用途**: 验证分布式架构在真实数据上工作

## 使用方法

### 快速测试（本地）
```bash
# Step 1: 测试基本集成
python test_distributed_explainer.py

# Step 2: 测试分布式架构（5 nodes, 2 workers）
python test_distributed_quick.py
```

### 完整基准测试（HPC）
```bash
# Submit Slurm job (100 nodes, 2/4/6/8/10 workers)
sbatch run_ogbn_distributed_bench.slurm

# Monitor progress
watch -n 60 'squeue -u $USER'

# Check logs
tail -f results/ogbn_distributed/*.log
```

### 可视化结果
```bash
python visualize_ogbn_distributed.py
```

## 技术细节

### 1. 解释器调用流程

```
Coordinator:
  1. 加载 OGBN-Papers100M 数据
  2. 采样 100 个测试节点
  3. 提取 2-hop 子图 (k_hop_subgraph)
  4. 创建 SubgraphTask (包含 node_id, subgraph_data, num_edges)
  5. 使用 min-heap 负载均衡分配任务到 workers

Workers (并行):
  1. 加载模型
  2. 初始化解释器 (HeuChase/ApxChase/GNNExplainer)
  3. 对每个任务:
     - 将 subgraph 移到 device
     - 运行解释: explainer._run(H=subgraph, root=target_node)
     - 记录 runtime 和解释结果
  4. 返回结果到 result_queue

Aggregator:
  1. 收集所有 worker 结果
  2. 计算统计: mean/std/min/max runtime
  3. 计算 speedup: worker_time_mean / parallel_time
  4. 保存结果到 JSON
```

### 2. 关键数据结构

#### SubgraphTask
```python
class SubgraphTask:
    task_id: int
    node_id: int              # 原始节点 ID
    subgraph_data: Data       # PyG Data object
    num_edges: int            # 用于负载均衡
```

#### Subgraph Data
```python
Data(
    x: [num_nodes, 128],      # 节点特征
    edge_index: [2, num_edges], # 边索引
    y: [num_nodes],           # 节点标签
    target_node: int,         # 目标节点（重标号后）
    subset: [num_nodes],      # 原始节点 ID 映射
)
```

### 3. 负载均衡

使用 min-heap 按边数分配任务：
- 大任务优先分配（sorted by edge count descending）
- 每次分配到当前负载最小的 worker
- 统计: mean/std/min/max load, balance ratio

### 4. 性能指标

```python
{
    'extraction_time': float,       # 子图提取时间
    'parallel_time': float,         # 并行执行时间
    'total_time': float,            # 总时间
    'task_runtime_mean': float,     # 平均任务时间
    'worker_time_mean': float,      # 平均 worker 时间
    'load_balance_ratio': float,    # 负载均衡比例
    'speedup': float,               # 加速比
}
```

## 依赖关系

### 必需文件
- `src/heuchase.py`: HeuChase implementation
- `src/apxchase.py`: ApxChase implementation
- `src/baselines.py`: GNNExplainer wrapper
- `src/matcher.py`: Constraint matching (可选，Sigma=None 时不用)
- `src/constraints.py`: Constraint definitions (可选)
- `src/Train_OGBN_HPC_MiniBatch.py`: GCN_2_OGBN model
- `models/OGBN_Papers100M_epoch_20.pth`: 训练好的模型

### Python 包
- torch
- torch_geometric
- ogb
- numpy
- heapq (标准库)
- multiprocessing (标准库)

## 已知限制和注意事项

### 1. Sigma=None 模式
- 当前使用 `Sigma=None`，不使用约束
- `gamma_fn` 会返回空集合 (early return)
- Coverage (`Sigma_star`) 为空
- 主要依赖 `verify_witness_fn` 来验证解释

### 2. Node Classification 特点
- 不是 graph classification
- 每个节点单独解释
- Subgraph extraction 自动处理节点重标号
- `target_node` 是重标号后的索引，不是原始 node_id

### 3. Memory 和性能
- 2-hop subgraph 可能有数千到数万边
- CPU-based processing (适合多核 HPC)
- 100 nodes × 3 explainers × 5 worker counts = 1500 tasks
- 预计运行时间: 数小时到十几小时

### 4. GNNExplainer 注意事项
- 需要训练 mask (100 epochs default)
- 比 chase-based 方法慢
- 可能需要调整 epochs/lr 获得好结果

## 下一步

### 1. 运行测试
```bash
# 本地快速测试
python test_distributed_quick.py
```

### 2. 提交完整基准测试
```bash
sbatch run_ogbn_distributed_bench.slurm
```

### 3. 分析结果
```bash
# 生成图表
python visualize_ogbn_distributed.py

# 查看结果文件
ls -lh results/ogbn_distributed/
cat results/ogbn_distributed/complete_results.pkl
```

### 4. 可能的优化
- 调整 explainer 参数 (k, B, m, epochs)
- 尝试不同的 num_hops (1, 2, 3)
- 实验不同的采样策略 (random vs degree-based)
- 添加更多性能指标 (explanation quality)

## 参考

- OGBN-Papers100M: https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M
- PyG k_hop_subgraph: https://pytorch-geometric.readthedocs.io/
- HeuChase paper: [Your paper reference]
- ApxChase implementation: src/apxchase.py
