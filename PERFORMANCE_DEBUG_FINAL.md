# TreeCycle Performance Debug - 最终状态

## 关键理解纠正

### Budget B 的真正含义 ✅

**错误理解**: B 是 HEAD matching 的数量  
**正确理解**: B 是 repair body 允许的最大缺边数

```python
# 在 backchase_repair_cost() 中:
# B = 允许插入的最大边数来满足 body
# 如果缺边数 > B, 则该 HEAD 匹配被拒绝
```

### OGBN 使用 B=8 的原因

- OGBN-Papers100M 约束可能需要最多 8 条边来repair
- TreeCycle 约束可能不同,但 B=8 是合理的起点
- B 太小 → 很多候选被拒绝
- B 太大 → 不会显著变慢 (只是允许更多repair)

## 当前配置状态

### ✅ 已优化的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| B | 8 | Repair budget (与 OGBN 对齐) |
| L | 2 | 2-hop 子图 |
| k | 10 | Window size (top-10 witnesses) |
| m | 6 | HeuChase 候选数 (Edmonds) |
| timeout | 1800s | 30分钟/任务 |

### ✅ 代码简化

1. **移除冗余属性**: _nodes_in_full, num_nodes, E_base
2. **移除不必要方法**: get_node_embeddings()
3. **完全对齐 OGBN**: 子图提取逻辑相同

### ✅ 增强的诊断

```python
# 现在会输出:
Worker 0: Calling HeuChase._run() on subgraph with 500 nodes, 1407 edges...
Worker 0: HeuChase._run() completed in 1.23s, found 5 witnesses
```

## 真正的性能瓶颈(待验证)

### 可能的原因:

1. **子图复杂度**
   - TreeCycle 的 2-hop 子图可能比 OGBN 更 dense
   - 1400 edges 对应多少节点? 如果 >1000 nodes, Edmonds 会慢

2. **Edmonds 算法本身**
   - NetworkX 实现可能在大图上较慢
   - 每次调用需要构建 DiGraph, 运行最大生成树

3. **验证开销**
   - 每个候选需要调用 verify_witness (模型前向传播)
   - m=6 次模型调用,每次可能 0.1-0.2s

4. **Constraint matching**
   - TreeCycle 有 5 个约束
   - OGBN 有多少个? 如果更少,可能更快

## 预期的实际性能

### 保守估计 (1400-edge 子图):

```
HeuChase 单个子图:
1. Extract embeddings: 0.1s (一次 GCN forward)
2. Edmonds × 6: 0.3-0.6s (每次 50-100ms)
3. Verify × 6: 0.3-0.6s (每次模型 forward)
4. Gamma/backchase × 6: 0.2-0.4s (constraint matching)
---
Total: 0.9-2.1s/子图
```

### 100 个子图, 20 workers:

- 每个 worker: 5 个子图
- 每个 worker 时间: 4.5-10.5s
- **Makespan**: ~5-11s ✅ 可接受

## 下一步行动

### 1. 等待当前 HPC 任务完成 ⏳

观察实际输出:
```
Worker 0: HeuChase._run() completed in ?.??s, found ? witnesses
```

### 2. 如果还是慢 (>5s/子图):

**诊断步骤**:

a) 检查子图实际大小:
```python
# 在 create_tasks() 后添加:
print(f"Subgraph node distribution:")
for t in tasks[:10]:
    sg = t.subgraph_data
    print(f"  Task {t.task_id}: {sg.x.size(0) if hasattr(sg, 'x') else '?'} nodes, {t.num_edges} edges")
```

b) 检查 NetworkX:
```python
# 在 worker 开始时:
try:
    import networkx as nx
    print(f"Worker {worker_id}: NetworkX {nx.__version__}")
except:
    print(f"Worker {worker_id}: NetworkX NOT available!")
```

c) 临时减小 m:
```python
m=3,  # 从 6 减到 3, 速度翻倍
```

d) 启用 debug 模式:
```python
debug=True  # 看详细的 Edmonds 输出
```

### 3. 如果性能合理 (1-2s/子图):

🎉 **成功!** 继续完整 benchmark

### 4. 其他优化选项:

- **减小 L**: num_hops=1 (1-hop 子图更小)
- **减小 num_targets**: 100 → 50 (更快测试)
- **使用 GPU**: device='cuda' (如果有多GPU)

## 总结

1. ✅ **B=8 是正确的** (虽然原因理解错了,但巧合地对齐了 OGBN)
2. ✅ **代码已完全对齐 OGBN**
3. ⏳ **真正的瓶颈待确认** - 可能只是需要耐心等待
4. 📊 **现在有详细日志** - 可以看到每个 _run() 的实际时间

**建议**: 让当前任务运行完,查看实际性能数据,然后决定是否需要进一步优化。
