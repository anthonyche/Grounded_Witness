# TreeCycle Distributed Benchmark - Hybrid CPU/GPU 策略

## 问题背景

之前的实现遇到 CUDA 多进程死锁：
- 20 个 worker 进程同时访问 1 个 GPU
- 所有数据（subgraph）都在 GPU 上
- 在 `HeuChase._run()` 调用后立即卡住

## 新策略：Hybrid CPU/GPU

### 核心思想
**数据在 CPU，模型在 GPU，推理时自动传输**

### 实现细节

1. **Worker 进程配置**：
   ```python
   data_device = 'cpu'          # 所有数据保持在 CPU
   model_device = 'cuda'         # 模型放在 GPU（如果可用）
   ```

2. **数据流**：
   ```
   Subgraph (CPU) → Model.forward() → 临时移到 GPU → 推理 → 返回 CPU
   ```

3. **关键修改**：
   ```python
   # 子图保持在 CPU
   subgraph = task.subgraph_data.to(data_device)  # CPU
   
   # 模型在 GPU
   model = model.to(model_device)  # CUDA
   
   # PyTorch Geometric 会自动处理数据传输
   # 当调用 model(subgraph.x, subgraph.edge_index) 时
   # GCNConv 内部会将输入移到模型所在的设备
   ```

### 优势

1. **避免 CUDA 死锁**：
   - 大部分数据操作在 CPU（无竞争）
   - GPU 只用于短暂的模型推理
   - 多进程不会同时持有 GPU 内存

2. **保持 GPU 加速**：
   - 模型推理（最耗时的部分）仍在 GPU
   - 验证步骤（constraint checking）仍快速

3. **内存安全**：
   - 20 个 workers 共享 128GB CPU 内存（足够）
   - GPU 内存压力小（只有模型参数）

### 与 OGBN 对比

| 项目 | OGBN | TreeCycle (新) |
|------|------|----------------|
| Workers | 10 | 20 |
| 数据位置 | CUDA | CPU |
| 模型位置 | CUDA | CUDA |
| 策略 | 全 GPU | Hybrid |

## 使用方法

### SLURM 脚本
保持原来的 GPU 配置：
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
```

### 运行
```bash
sbatch run_treecycle_distributed_bench.slurm
```

### 输出日志示例
```
Worker 1: Started, model_device=cuda, data_device=cpu, tasks=5
Worker 1: Model loaded and ready on cuda
Worker 1: Data will stay on cpu (避免 CUDA 多进程冲突)
Worker 1: Subgraph on cpu - nodes=748, edges=1496, target=374
Worker 1: Calling heuchase._run()...
```

## 技术细节

### PyTorch Geometric 的自动设备管理

GCNConv 的 forward 方法会处理设备不匹配：
```python
class GCNConv:
    def forward(self, x, edge_index):
        # PyG 内部会检查 x 和 self.weight 的设备
        # 如果不匹配，会自动将 x 移到 self.weight 所在设备
        # 推理完成后，输出在模型设备上
```

### HeuChase/ApxChase/ExhaustChase

这些算法会：
1. 接收 CPU 上的 subgraph
2. 调用 `model(x, edge_index)` 时，PyG 自动处理传输
3. 获得 GPU 推理结果（自动移回 CPU）
4. 在 CPU 上进行 constraint checking

### 性能预期

- **模型推理**：GPU 加速（~5-10x 比 CPU 快）
- **数据传输开销**：可忽略（subgraph 很小，几百节点）
- **整体性能**：接近全 GPU，但稳定性更好

## 故障排除

### 如果仍然卡住
1. 检查是否真的在用 CPU 存数据：
   ```bash
   grep "data_device=cpu" logs/*.out
   ```

2. 检查 GPU 内存：
   ```bash
   nvidia-smi
   ```

3. 如果 GPU 内存爆满，回退到纯 CPU：
   ```python
   DEVICE = 'cpu'  # 在 main() 中修改
   ```

### 如果需要纯 CPU 测试
运行：
```bash
sbatch run_treecycle_cpu_test.slurm
```

## 下一步

如果这个策略成功：
- ✅ 验证 HeuChase 不再卡住
- ✅ 比较 CPU vs Hybrid GPU 的性能
- ✅ 扩展到所有 explainers (ApxChase, ExhaustChase, GNNExplainer, PGExplainer)
- ✅ 完整 benchmark 运行
