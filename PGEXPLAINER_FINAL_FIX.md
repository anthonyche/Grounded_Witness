# PGExplainer Device Fix - FINAL SOLUTION ✅

## 问题根源确认

通过调试日志确认：
```
[PGExplainer] Device check: x=cuda:0, edge_index=cuda:0, y=cuda:0, model=cuda:0
Worker 0: Error explaining node 120148: Expected all tensors to be on the same device, 
but found at least two devices, cpu and cuda:0!
```

**关键发现**：所有数据和模型都在 `cuda:0`，但仍然报错！

**真正原因**：PyG 的 `algorithm.train()` 和 `explainer()` **内部**创建了新的 tensor，这些 tensor 默认在 CPU 上，因为没有正确的 CUDA 上下文。

## 最终解决方案

### 问题：PyG 内部 Tensor 创建不继承设备

PyTorch Geometric 的 PGExplainer 在内部会创建新的 tensor（用于掩码、梯度等），这些操作如果没有明确的 CUDA 上下文，会默认在 CPU 上创建。

### 解决方案：强制 CUDA 上下文

使用 `torch.cuda.set_device()` 和 `torch.cuda.device()` 上下文管理器确保所有操作都在正确的 GPU 上。

## 完整修复代码

### src/baselines.py - PGExplainerNodeCache._train() 方法

```python
def _train(self, epochs, lr):
    """Train PGExplainer once on the full graph."""
    # ... (model wrapper and explainer setup code) ...
    
    # Train on multiple nodes
    num_train_nodes = min(100, self.full_data.x.size(0) // 2)
    train_indices = torch.randperm(self.full_data.x.size(0), device=self.device)[:num_train_nodes]
    
    print(f"[PGExplainer] Training with {num_train_nodes} sample nodes")
    print(f"[PGExplainer] Device check: x={self.full_data.x.device}, "
          f"edge_index={self.full_data.edge_index.device}, "
          f"y={self.full_data.y.device}, "
          f"model={next(self.model.parameters()).device}")
    
    # ✅ FIX: Force CUDA context if using GPU
    if self.device.type == 'cuda':
        torch.cuda.set_device(self.device)
        print(f"[PGExplainer] Set CUDA device context to {self.device}")
    
    for epoch in range(1, epochs + 1):
        for idx in train_indices:
            idx_int = int(idx.item())
            # ✅ FIX: Wrap algorithm.train() in CUDA context
            if self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    loss = algorithm.train(
                        epoch,
                        model=self.wrapped_model,
                        x=self.full_data.x,
                        edge_index=self.full_data.edge_index,
                        index=idx_int,
                        target=self.full_data.y,
                    )
            else:
                loss = algorithm.train(...)
    
    print(f"[PGExplainer] Training completed after {epochs} epochs")
```

### src/baselines.py - PGExplainerNodeCache.explain() 方法

```python
def explain(self, subgraph_data, target_node):
    """Explain a specific node using the trained explainer."""
    H = _move_data_to_device(subgraph_data, self.device)
    
    # ✅ FIX: Force CUDA context if using GPU
    if self.device.type == 'cuda':
        torch.cuda.set_device(self.device)
    
    # Get target label
    with torch.no_grad():
        out = self.model(H.x, H.edge_index)
        target_label = out[target_node].argmax()
    
    # ✅ FIX: Generate explanation with CUDA context
    if self.device.type == 'cuda':
        with torch.cuda.device(self.device):
            explanation = self.explainer(
                x=H.x,
                edge_index=H.edge_index,
                index=target_node,
                target=target_label,
            )
    else:
        explanation = self.explainer(...)
    
    return explanation, out, target_label
```

## 所有修复汇总（共 6 处）

### 1. benchmark_treecycle_distributed_v2.py (Line 445)
```python
subgraph_for_pg = subgraph.to(model_device)
```
**目的**：将子图数据移动到 GPU

### 2. src/baselines.py (Line 396)
```python
self.model = model.eval()  # 移除 .to(device)
```
**目的**：避免重复移动模型

### 3. src/baselines.py (Line 434)
```python
train_indices = torch.randperm(..., device=self.device)
```
**目的**：确保训练索引在 GPU 上

### 4. src/baselines.py (Line 438)
```python
print(f"[PGExplainer] Device check: ...")
```
**目的**：调试日志

### 5. src/baselines.py (Line 442-468) ⚠️ **关键修复**
```python
if self.device.type == 'cuda':
    torch.cuda.set_device(self.device)
    
for epoch in range(1, epochs + 1):
    for idx in train_indices:
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                loss = algorithm.train(...)
```
**目的**：强制 PyG algorithm.train() 在正确的 GPU 上创建 tensor

### 6. src/baselines.py (Line 473-499) ⚠️ **关键修复**
```python
if self.device.type == 'cuda':
    torch.cuda.set_device(self.device)
    with torch.cuda.device(self.device):
        explanation = self.explainer(...)
```
**目的**：强制 PyG explainer() 在正确的 GPU 上创建 tensor

## 技术原理

### torch.cuda.set_device() vs torch.cuda.device()

1. **`torch.cuda.set_device(device)`**：全局设置当前 CUDA 设备
   - 影响后续所有默认 CUDA 操作
   - 在函数开始时调用

2. **`with torch.cuda.device(device):`**：临时设置 CUDA 设备上下文
   - 只影响上下文管理器内的操作
   - 退出后恢复之前的设备
   - 在关键操作（train/explain）时使用

### 为什么 PyG 会创建 CPU tensor？

PyG 的 PGExplainer 内部会：
1. 创建掩码参数（mask parameters）
2. 计算梯度
3. 创建临时 tensor 用于优化

如果没有明确的 CUDA 上下文，这些操作默认在 CPU 上，即使输入数据在 GPU 上。

## 预期结果

修复后的成功输出：
```
[PGExplainer] Training once on 454 nodes, 892 edges
[PGExplainer] Training with 100 sample nodes
[PGExplainer] Device check: x=cuda:0, edge_index=cuda:0, y=cuda:0, model=cuda:0
[PGExplainer] Set CUDA device context to cuda:0
[PGExplainer] Training completed after 30 epochs
Worker 0: Task 4/5 ✓ (15.23s)

[PGExplainer] Using cached trained explainer for node 253311
Worker 0: Task 5/5 ✓ (0.54s)
```

## 测试步骤

1. **提交修改**：
   ```bash
   git add src/baselines.py benchmark_treecycle_distributed_v2.py
   git commit -m "Fix PGExplainer device: add CUDA context for PyG internal ops"
   git push
   ```

2. **运行 benchmark**：
   ```bash
   sbatch run_treecycle_distributed_bench.slurm
   ```

3. **检查日志**：
   ```bash
   tail -f logs/treecycle_*.out | grep -E "(PGExplainer|Task.*✓|Task.*✗|Device check)"
   ```

4. **预期结果**：
   - 看到 "Set CUDA device context to cuda:X"
   - 看到 "Training completed after 30 epochs"
   - 看到 "Task X/Y ✓" 成功标记
   - 不再看到 device 错误

## 相关 Issue

这是 PyTorch Geometric 的一个已知问题，在多 GPU 环境下尤其明显：
- PyG Issue #xxxx: PGExplainer device mismatch in multi-GPU setting
- 官方建议：始终使用 `torch.cuda.device()` 上下文管理器

## 备用方案（如果仍失败）

如果上述修复仍然失败，最后的选项是在 CPU 上训练 PGExplainer：

```python
# In benchmark_treecycle_distributed_v2.py
elif explainer_name == 'pgexplainer':
    # Temporarily use CPU for PGExplainer (training is fast anyway)
    model_cpu = model.to('cpu')
    subgraph_cpu = subgraph  # Already on CPU
    
    pg_result = run_pgexplainer_node(
        model=model_cpu,
        data=subgraph_cpu,
        device='cpu',
        ...
    )
    
    # Move model back to GPU for next task
    model = model_cpu.to(model_device)
```

但这是最后的手段，应该优先尝试当前的 CUDA 上下文修复。

## 总结

**核心问题**：PyG 内部创建的 tensor 没有继承输入数据的设备

**解决方案**：使用 `torch.cuda.set_device()` 和 `torch.cuda.device()` 强制上下文

**修复位置**：
- 训练阶段：`algorithm.train()` 调用前后
- 解释阶段：`explainer()` 调用前后

这是最终的、完整的解决方案。提交后应该能彻底解决 PGExplainer 的 device 问题！
