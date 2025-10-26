# PGExplainer Device Fix - Final Solution

## 问题描述

在 HPC 4-GPU 环境运行时，PGExplainer 持续报错：

```
Worker 3: Error explaining node 709271: Expected all tensors to be on the same device, 
but found at least two devices, cpu and cuda:3!
```

## 根本原因：双重 Device 问题

### 问题 1：数据未移动到 GPU ✅ 已修复
- **位置**：`benchmark_treecycle_distributed_v2.py` 第 445 行
- **原因**：传递给 `run_pgexplainer_node` 的 `subgraph` 在 CPU 上
- **修复**：添加 `subgraph_for_pg = subgraph.to(model_device)`

### 问题 2：模型重复移动 ⚠️ **关键问题**
- **位置**：`src/baselines.py` 第 396 行 `PGExplainerNodeCache.__init__`
- **原因**：`self.model = model.to(device).eval()` 
  - 模型已经在 `cuda:3` 上
  - 再次调用 `.to(cuda:3)` 可能创建新的参数副本
  - 破坏模型状态一致性
- **修复**：移除 `.to(device)` 调用

## 完整修复代码

### 修复 1：benchmark_treecycle_distributed_v2.py (Line 440-461)

```python
elif explainer_name == 'pgexplainer':
    # Import run_pgexplainer_node inside worker
    from baselines import run_pgexplainer_node
    
    # ✅ FIX: Move subgraph to model's device for PGExplainer training
    subgraph_for_pg = subgraph.to(model_device)
    
    # Use cached PGExplainer (trains once on first call per worker)
    pg_result = run_pgexplainer_node(
        model=model,
        data=subgraph_for_pg,  # Data on same device as model
        target_node=int(target_node),
        epochs=explainer_config.get('train_epochs', 30),
        lr=explainer_config.get('train_lr', 0.003),
        device=model_device,  # Use model's device
        use_cache=True,  # Enable caching to avoid retraining
    )
    explanation_result = {
        'edge_mask': pg_result.get('edge_mask'),
        'pred': pg_result.get('pred'),
        'success': pg_result.get('edge_mask') is not None
    }
```

### 修复 2：src/baselines.py (Line 393-403) ⚠️ **关键修复**

**修改前**（错误）：
```python
class PGExplainerNodeCache:
    """Cache for trained PGExplainer to avoid retraining for each node."""
    
    def __init__(self, model, full_data, device, epochs=30, lr=0.003):
        self.model = model.to(device).eval()  # ❌ 重复移动模型
        self.full_data = _move_data_to_device(full_data, device)
        self.device = device
        self.explainer = None
        self.wrapped_model = None
        self._train(epochs, lr)
```

**修改后**（正确）：
```python
class PGExplainerNodeCache:
    """Cache for trained PGExplainer to avoid retraining for each node."""
    
    def __init__(self, model, full_data, device, epochs=30, lr=0.003):
        # ✅ Don't call model.to(device) - model already on correct device
        self.model = model.eval()  # Only set eval mode
        self.full_data = _move_data_to_device(full_data, device)
        self.device = device
        self.explainer = None
        self.wrapped_model = None
        self._train(epochs, lr)
```

## 技术分析

### 为什么 model.to(device) 会导致问题？

1. **传入的 model 已经在正确设备**：
   ```python
   # In worker_process:
   model = GCN(...).to(model_device)  # Already on cuda:3
   ```

2. **重复调用 .to() 的副作用**：
   ```python
   model.to(device)  # device = cuda:3, but model already on cuda:3
   ```
   - PyTorch 可能创建新的参数对象（即使 device 相同）
   - 破坏参数引用
   - 导致训练时 device 不一致

3. **正确做法**：
   - 信任传入的 model 已经在正确设备上
   - 只调用 `.eval()` 设置评估模式
   - 不要再次移动设备

### Device 流程验证

```
Worker 3 启动:
├─ model = GCN(...).to('cuda:3')          # ✓ Model on cuda:3
├─ subgraph = task.subgraph_data.to('cpu')  # ✓ Subgraph on CPU
│
├─ PGExplainer 执行:
│  ├─ subgraph_for_pg = subgraph.to('cuda:3')  # ✓ Move to cuda:3
│  ├─ run_pgexplainer_node(model, subgraph_for_pg, device='cuda:3')
│  │
│  └─ PGExplainerNodeCache.__init__:
│     ├─ self.model = model.eval()  # ✅ Keep on cuda:3 (NEW FIX)
│     │   (NOT model.to(device) - REMOVED!)
│     ├─ self.full_data = subgraph_for_pg.to('cuda:3')  # ✓ Already on cuda:3
│     │
│     └─ _train():
│        └─ algorithm.train(x=self.full_data.x, ...)  # ✓ All on cuda:3
```

## 预期结果

修复后的成功输出：
```
Worker 3: Running pgexplainer on subgraph (nodes=485, edges=946, target=421)...
[PGExplainer] Training new explainer (will be cached)
[PGExplainer] Training once on 485 nodes, 946 edges
[PGExplainer] Training with 100 sample nodes
[PGExplainer] Training completed after 30 epochs
Worker 3: Task 2/5 ✓ (15.23s)

[PGExplainer] Using cached trained explainer for node 378556
Worker 3: Task 3/5 ✓ (0.54s)  # Cached - much faster!
```

## 文件修改清单

1. ✅ **src/baselines.py**
   - Line 396: 移除 `model.to(device)`
   - 修改：`self.model = model.eval()` (不再调用 .to())

2. ✅ **benchmark_treecycle_distributed_v2.py**  
   - Line 445: 添加 `subgraph_for_pg = subgraph.to(model_device)`
   - Line 449: 使用 `data=subgraph_for_pg`

## 验证步骤

在 HPC 上运行：
```bash
sbatch run_treecycle_distributed_bench.slurm
```

检查输出：
```bash
tail -f logs/treecycle_*.out | grep -E "(Worker|PGExplainer|Task|Error)"
```

预期：
- ✅ 不再出现 "Expected all tensors to be on the same device" 错误
- ✅ 首次训练 ~15-20秒
- ✅ 缓存命中后 <1秒

## 总结

**两处关键修复**：
1. 数据移动：`subgraph.to(model_device)` before calling PGExplainer
2. 模型保持：不要在 `PGExplainerNodeCache` 中再次调用 `model.to(device)`

**核心原则**：
- 模型由 worker 创建，已在正确 GPU 上
- 数据需要显式移动到模型所在 GPU
- 不要重复移动已经在正确设备上的模型
