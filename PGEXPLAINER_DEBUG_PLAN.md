# PGExplainer Device 问题深度调试

## 当前状态

在 HPC 上运行时，PGExplainer 持续报错：
```
Worker 3: Error explaining node 614202: Expected all tensors to be on the same device, 
but found at least two devices, cpu and cuda:3!
```

## 已完成的修复（3 处）

### 修复 1: benchmark_treecycle_distributed_v2.py (Line 445)
```python
# ✅ 移动子图到 GPU
subgraph_for_pg = subgraph.to(model_device)
pg_result = run_pgexplainer_node(model=model, data=subgraph_for_pg, ...)
```

### 修复 2: src/baselines.py (Line 396)
```python
# ✅ 移除重复的 model.to(device)
def __init__(self, model, full_data, device, epochs=30, lr=0.003):
    self.model = model.eval()  # 不再调用 .to(device)
    self.full_data = _move_data_to_device(full_data, device)
```

### 修复 3: src/baselines.py (Line 434) - NEW
```python
# ✅ train_indices 也要在 GPU 上
train_indices = torch.randperm(self.full_data.x.size(0), device=self.device)[:num_train_nodes]
```

### 修复 4: src/baselines.py (Line 438) - NEW (调试信息)
```python
# ✅ 添加设备检查日志
print(f"[PGExplainer] Device check: x={self.full_data.x.device}, "
      f"edge_index={self.full_data.edge_index.device}, "
      f"y={self.full_data.y.device}, "
      f"model={next(self.model.parameters()).device}")
```

## 问题可能的根源

### 猜测 1: PyG PGExplainer.train() 内部创建了 CPU tensor
`algorithm.train()` 是 PyG 库的方法，可能在内部：
- 创建了新的 tensor（默认在 CPU）
- 使用了 numpy 数组然后转换
- 有某些操作没有正确继承 device

### 猜测 2: target 参数格式问题
```python
target=self.full_data.y  # 传递整个 y 向量
```
可能需要：
```python
target=self.full_data.y[idx_int]  # 只传递单个节点的标签
```

### 猜测 3: wrapped_model 的问题
```python
class NodeModelWrapper(torch.nn.Module):
    def forward(self, x, edge_index, **kwargs):
        return self.model(x, edge_index)
```
可能在某些情况下丢失了 device 信息。

## 下一步调试策略

### 步骤 1: 运行带调试信息的版本
提交当前修改到 HPC，查看日志中的设备信息：
```
[PGExplainer] Device check: x=cuda:3, edge_index=cuda:3, y=cuda:3, model=cuda:3
```

如果所有设备都是 cuda:3，但仍然报错，说明问题在 `algorithm.train()` 内部。

### 步骤 2: 检查 PyG 版本和已知问题
```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

查看是否是 PyG 的已知 bug。

### 步骤 3: 尝试替代方案
如果 PyG 的 PGExplainer 有 device 问题，考虑：

**方案 A: 强制所有操作在同一设备**
```python
def _train(self, epochs, lr):
    # 确保 algorithm 也知道设备
    algorithm = PGExplainer(epochs=epochs, lr=lr)
    
    # 可能需要手动设置 algorithm 的设备
    if hasattr(algorithm, 'to'):
        algorithm = algorithm.to(self.device)
```

**方案 B: 使用 torch.cuda.set_device**
```python
def _train(self, epochs, lr):
    # 强制设置当前 CUDA 设备
    if self.device.type == 'cuda':
        torch.cuda.set_device(self.device)
    
    # ... training code ...
```

**方案 C: 包装 algorithm.train 调用**
```python
def _train(self, epochs, lr):
    for epoch in range(1, epochs + 1):
        for idx in train_indices:
            with torch.cuda.device(self.device):  # 上下文管理器
                loss = algorithm.train(...)
```

**方案 D: 最保守 - 暂时在 CPU 上训练**
```python
# 临时解决方案：在 CPU 上训练 PGExplainer
# 训练很快，不会成为瓶颈
if explainer_name == 'pgexplainer':
    subgraph_for_pg = subgraph  # 保持在 CPU
    pg_result = run_pgexplainer_node(
        model=model.to('cpu'),  # 临时移到 CPU
        data=subgraph_for_pg,
        device='cpu',
    )
    model.to(model_device)  # 训练后移回 GPU
```

## 立即执行的操作

1. **提交当前修改**到 HPC：
   - 3 处 device 修复
   - 1 处调试日志

2. **查看日志输出**：
   ```bash
   sbatch run_treecycle_distributed_bench.slurm
   tail -f logs/treecycle_*.out | grep -E "(PGExplainer|Device check|Error)"
   ```

3. **根据日志决定**：
   - 如果设备都是 cuda:3 → 问题在 PyG 内部 → 尝试方案 A/B/C
   - 如果设备有 CPU → 找到哪个 tensor 在 CPU → 针对性修复
   - 如果无法解决 → 使用方案 D（CPU 训练）

## 文件清单

修改的文件：
1. ✅ `src/baselines.py` (3 处修改)
   - Line 396: 移除 `model.to(device)`
   - Line 434: `train_indices` 使用 `device=self.device`
   - Line 438: 添加设备检查日志

2. ✅ `benchmark_treecycle_distributed_v2.py` (1 处修改)
   - Line 445: `subgraph_for_pg = subgraph.to(model_device)`

## 预期日志输出

**成功情况**：
```
[PGExplainer] Training once on 396 nodes, 774 edges
[PGExplainer] Training with 100 sample nodes
[PGExplainer] Device check: x=cuda:3, edge_index=cuda:3, y=cuda:3, model=cuda:3
[PGExplainer] Training completed after 30 epochs
Worker 3: Task 4/5 ✓ (15.23s)
```

**失败情况（需要更多调试）**：
```
[PGExplainer] Device check: x=cuda:3, edge_index=cuda:3, y=cuda:3, model=cuda:3
Worker 3: Error explaining node 614202: Expected all tensors to be on the same device...
```
→ 说明问题在 `algorithm.train()` 内部

## 紧急后备方案

如果所有方案都失败，可以暂时禁用 PGExplainer：
```python
# In benchmark_treecycle_distributed_v2.py
EXPLAINERS = ['heuchase', 'apxchase', 'exhaustchase']  # 移除 'pgexplainer'
```

先完成其他解释器的 benchmark，PGExplainer 可以后续单独调试。
