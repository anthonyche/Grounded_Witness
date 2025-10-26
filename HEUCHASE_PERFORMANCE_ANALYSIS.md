# HeuChase 性能分析与优化建议

## 问题诊断

### 观察到的现象
```
Worker 1: Task 1/5 (node 24660, 1407 edges)...
Worker 1: Subgraph moved to device, running explainer...
# 然后卡住,没有后续输出
```

### 根本原因

HeuChase **确实使用 Edmonds 算法**,但在大子图上仍然较慢,原因有:

#### 1. **Edmonds 算法本身的复杂度**
```python
# 在 _candidate_by_edmonds() 中:
# - 构建 NetworkX DiGraph: O(E)
# - 每条无向边变成2条有向边: 2×1407 = 2814 条边
# - 运行 nx.maximum_spanning_arborescence: O(E log V)
# - 对于 1407 edges, ~500 nodes: 大约 1407 × log(500) ≈ 12,000 操作
```

#### 2. **每次生成候选都要运行模型**
```python
# 在 HeuChase._run() 中:
for t in range(self.m):  # m=6 次
    edge_mask = _candidate_by_edmonds(H, root, emb, noise_std)
    Gs = _induce_subgraph_from_edges(H, edge_mask)
    ok = self.verify_witness_fn(self.model, root, Gs)  # 模型前向传播!
```

每个子图需要:
- 6次 Edmonds 算法调用
- 6次模型前向传播 (verify_witness)
- 6次 Gamma 计算 (检查constraint grounding)

#### 3. **Embedding 提取可能重复**
```python
# _extract_node_embeddings() 在每次调用时:
emb = _extract_node_embeddings(self.model, H)  # 每个子图调用一次,OK
# 但之前 GCN 没有 get_node_embeddings() 方法,会 fallback 到运行完整前向传播!
```

## 修复措施

### 已实现 ✅

1. **添加 GCN.get_node_embeddings() 方法**
   ```python
   def get_node_embeddings(self, data):
       with torch.no_grad():
           x = self.conv1(data.x, data.edge_index)
           x = F.relu(x)
           return x
   ```
   - 避免运行第二层 GCN
   - 直接返回第一层的 embeddings

2. **减小 Budget 参数**
   ```python
   B=50  # 从 100 减到 50
   ```
   - 减少 Gamma 计算的 HEAD matching 数量

3. **减小 ExhaustChase 迭代次数**
   ```python
   max_enforce_iterations=50  # 从 100 减到 50
   ```

4. **添加详细的性能日志**
   ```python
   print(f"Worker {worker_id}: Calling {explainer_name}._run() ...")
   run_start = time.time()
   Sigma_star, S_k = explainer._run(H=subgraph, root=int(target_node))
   run_elapsed = time.time() - run_start
   print(f"Worker {worker_id}: {explainer_name}._run() completed in {run_elapsed:.2f}s")
   ```

5. **优化 signal handler**
   ```python
   # 在循环外设置一次,而不是每个任务都重新设置
   signal.signal(signal.SIGALRM, timeout_handler)
   ```

### 预期性能

#### 对于 1400-edge 子图:

| 组件 | 估计时间 | 说明 |
|------|----------|------|
| Embedding提取 | ~0.05s | 一次GCN forward (第一层) |
| Edmonds (×6) | ~0.3-0.6s | 6次最大生成树,每次 ~50-100ms |
| Verify (×6) | ~0.3s | 6次模型前向传播 |
| Gamma (×6) | ~0.2-0.4s | 6次 HEAD matching (B=50) |
| **Total** | **~1-2s** | 每个子图 |

#### 对于 100 个子图, 20 workers:

- 每个 worker: 5 个子图
- 每个子图: ~1-2s
- 每个 worker 总时间: ~5-10s
- **Makespan (并行)**: ~5-10s ✅

## 为什么还是慢?

### 可能的原因:

1. **NetworkX 没有安装或导入失败**
   - 检查 `_HAS_NX` 是否为 True
   - 如果 False, 会 fallback 到更慢的贪心算法

2. **子图比预期更大**
   - 1407 edges 可能对应 >500 nodes
   - TreeCycle 的 2-hop 子图可能非常dense

3. **模型在 CPU 上运行很慢**
   - workers 使用 CPU (device=cpu)
   - GCN 在 CPU 上比 GPU 慢 10-50倍

4. **实际上没有卡住,只是很慢**
   - 需要等待 1-2分钟才能看到结果
   - 添加的详细日志还没输出

## 建议的调试步骤

### 1. 检查 NetworkX
```python
# 在 worker 开始时打印
import networkx as nx
print(f"NetworkX available: {nx is not None}")
print(f"NetworkX version: {nx.__version__}")
```

### 2. 减小 m 参数 (临时测试)
```python
explainer = HeuChase(
    ...,
    m=3,  # 从 6 减到 3,速度翻倍
    ...
)
```

### 3. 使用更小的测试集
```python
# 在 main() 中:
num_targets=10,  # 从 100 减到 10
```

### 4. 启用 debug 模式查看详细进度
```python
explainer = HeuChase(
    ...,
    debug=True  # 从 False 改为 True
)
```

### 5. 检查实际子图大小
```python
# 在 coordinator.create_tasks() 后添加:
print(f"Subgraph stats:")
print(f"  Nodes: min={min(t.subgraph_data.num_nodes for t in tasks)}, "
      f"max={max(t.subgraph_data.num_nodes for t in tasks)}, "
      f"mean={np.mean([t.subgraph_data.num_nodes for t in tasks]):.1f}")
```

## 最终建议

### 短期 (立即实施):

1. ✅ **等待更长时间** - HeuChase 在 1400-edge 子图上可能需要 1-2分钟/任务
2. ✅ **查看完整日志** - 新的详细日志会显示每个 _run() 的实际时间
3. **减小 m 参数** - 如果太慢,可以临时改为 m=3

### 中期 (优化):

1. **使用 GPU** - 将 workers 改为使用 GPU (如果有多个 GPU)
2. **减小子图范围** - 改为 1-hop 而不是 2-hop (num_hops=1)
3. **Pre-compute embeddings** - 在 coordinator 阶段提取所有 embeddings

### 长期 (重构):

1. **并行化 Edmonds** - 使用多线程生成多个候选
2. **Caching** - 缓存已见过的子图的结果
3. **Early stopping** - 如果 coverage 饱和就提前停止

## 当前状态

✅ 代码已修复:
- GCN.get_node_embeddings() 添加
- Budget 减小到 50
- Signal handler 优化
- 详细性能日志添加

⏳ 需要验证:
- HeuChase 实际运行时间
- 是否真的卡住,还是只是慢
- NetworkX 是否正常工作

建议: **等待当前任务完成**,查看新的详细日志输出来确认实际性能。
