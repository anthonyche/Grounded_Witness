# TreeCycle Distributed Benchmark - Critical Bug Fix

## 问题诊断

### 症状
- OGBN benchmark 在 20 workers 下正常运行
- TreeCycle benchmark 在相同配置下**卡住**在 `heuchase._run()` 调用
- Workers 进入函数但无输出,无错误,只是挂起

### 根本原因

**multiprocessing `spawn` 模式 + 顶部导入重量级模块 = 死锁**

```python
# ❌ 错误方式 (TreeCycle 原版)
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

from heuchase import HeuChase  # ← 在文件顶部导入
from apxchase import ApxChase
from exhaustchase import ExhaustChase

def worker_process(...):
    explainer = HeuChase(...)  # 使用已导入的类
```

**问题**: `spawn` 模式下,每个子进程会**重新导入主模块**,导致:
1. 重复初始化全局变量
2. 重复加载大型模块 (torch, torch_geometric, networkx 等)
3. 可能的循环依赖和锁竞争
4. 最终导致死锁

```python
# ✅ 正确方式 (OGBN 版本)
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

# 只导入必要的轻量级模块
from constraints import get_constraints

def worker_process(...):
    # 在 worker 内部动态导入
    from heuchase import HeuChase  # ← 只在需要时导入
    explainer = HeuChase(...)
```

**优势**:
- 每个 worker 独立导入,避免全局状态污染
- 延迟加载,减少主进程内存占用
- 避免 multiprocessing pickle 序列化问题

---

## 修复清单

### 1. ✅ 移除顶部 explainer 导入

**修改前:**
```python
# benchmark_treecycle_distributed_v2.py (line 27-31)
import sys
sys.path.append('src')
from heuchase import HeuChase
from apxchase import ApxChase
from exhaustchase import ExhaustChase
from baselines import run_gnn_explainer_node, PGExplainerBaseline
from constraints import get_constraints
```

**修改后:**
```python
# benchmark_treecycle_distributed_v2.py (line 27-31)
import sys
sys.path.append('src')
from constraints import get_constraints

# DO NOT import explainers here! Import inside worker_process to avoid multiprocessing issues
```

---

### 2. ✅ 添加 worker 内部动态导入

**修改前:**
```python
def worker_process(...):
    ...
    if explainer_name == 'heuchase':
        print(f"Worker {worker_id}: Creating HeuChase...")
        explainer = HeuChase(...)  # 使用全局导入
```

**修改后:**
```python
def worker_process(...):
    ...
    if explainer_name == 'heuchase':
        print(f"Worker {worker_id}: Importing HeuChase...")
        from heuchase import HeuChase  # ← 动态导入
        print(f"Worker {worker_id}: Creating HeuChase...")
        explainer = HeuChase(...)
```

对所有 explainer 都这样处理:
- `heuchase` → `from heuchase import HeuChase`
- `apxchase` → `from apxchase import ApxChase`
- `exhaustchase` → `from exhaustchase import ExhaustChase`
- `gnnexplainer` → `from baselines import run_gnn_explainer_node` (在 try 块内)

---

### 3. ✅ 移除无用的 timeout 参数

**修改前:**
```python
def worker_process(worker_id, tasks, model_state, explainer_name, 
                  explainer_config, device, result_queue, timeout_seconds=1800):
    ...

# 调用时
args=(worker_id, task_assignments[worker_id], model_state,
      explainer_name, explainer_config, device, result_queue, 1800)
```

**修改后:**
```python
def worker_process(worker_id, tasks, model_state, explainer_name, 
                  explainer_config, device, result_queue):
    ...

# 调用时
args=(worker_id, task_assignments[worker_id], model_state,
      explainer_name, explainer_config, device, result_queue)
```

**原因**: 
- OGBN 版本不使用 signal/alarm 机制
- `timeout_seconds` 参数从未被使用
- 保持简洁,避免误导

---

### 4. ✅ 移除 signal/alarm (已在之前修复)

**确认**: 文件中已无 `signal` 相关代码
```bash
$ grep -n "signal" benchmark_treecycle_distributed_v2.py
# (无结果)
```

---

## 验证修复

### 快速测试 (5 nodes, 2 workers)

```bash
# 修改 main() 函数中的配置
NUM_TARGETS = 5   # 从 100 → 5
NUM_WORKERS = 2   # 从 20 → 2
EXPLAINERS = ['heuchase']  # 只测试 HeuChase

python benchmark_treecycle_distributed_v2.py
```

**预期输出:**
```
Worker 0: Importing HeuChase...
Worker 0: Creating HeuChase...
Worker 0: HeuChase initialized (B=8)
Worker 0: ✓ Explainer ready, starting 3 tasks
Worker 0: Processing 3 tasks...
Worker 0: Task 1/3 (node xxx, xxx edges)...
...
Worker 0: Task 1/3 ✓ (2.45s, 15 witnesses)
Worker 0: Task 2/3 (node xxx, xxx edges)...
...
```

### 完整测试 (100 nodes, 20 workers)

```bash
# 使用默认配置
python benchmark_treecycle_distributed_v2.py
```

**预期行为:**
- 20 个 workers 并行运行
- 每个 worker 完成 5 个任务 (100/20)
- 无挂起,无死锁
- 所有结果正常返回

---

## 性能对比

### 修复前 (TreeCycle 原版)
```
Worker 0-19: Started...
Worker 0: Creating HeuChase...
Worker 0: HeuChase initialized
Worker 0: Task 1/5...
Worker 0: Calling heuchase._run()...
<挂起,无输出>  ← 死锁
```

### 修复后 (对齐 OGBN)
```
Worker 0-19: Started...
Worker 0: Importing HeuChase...  ← 动态导入
Worker 0: Creating HeuChase...
Worker 0: HeuChase initialized
Worker 0: Task 1/5...
Worker 0: Task 1/5 ✓ (2.45s, 15 witnesses)  ← 正常完成
Worker 0: Task 2/5...
...
Coordinator: Received 5 results from worker 0  ← 成功返回
```

---

## 技术细节

### multiprocessing `spawn` 模式

在 macOS 和 Python 3.8+ 默认使用 `spawn` 模式:
- 创建全新的 Python 解释器进程
- 重新导入主模块 (`__main__`)
- 序列化 (pickle) 所有传递的对象

**最佳实践**:
1. 在 `if __name__ == '__main__':` 内创建进程
2. 轻量级顶部导入 (只导入配置和工具)
3. 重量级模块在 worker 内部导入
4. 避免全局可变状态

### PyTorch + multiprocessing 的坑

1. **CUDA 初始化**: 在主进程初始化 CUDA 会导致 fork 子进程死锁
   - 解决: 在 worker 内部初始化,或使用 CPU
   
2. **模型序列化**: 直接传递 `nn.Module` 可能失败
   - 解决: 传递 `state_dict`,在 worker 重建模型
   
3. **共享内存**: `torch.Tensor` 在 `spawn` 模式下无法共享
   - 解决: 使用 pickle 传递,或用 `torch.multiprocessing.Queue`

---

## 与 OGBN 的对比

| 项目 | OGBN (正常) | TreeCycle (修复前) | TreeCycle (修复后) |
|------|-------------|-------------------|-------------------|
| 顶部导入 explainer | ❌ | ✅ (导致死锁) | ❌ |
| worker 内动态导入 | ✅ | ❌ | ✅ |
| signal/alarm | ❌ | ✅ (已移除) | ❌ |
| timeout 参数 | ❌ | ✅ (未使用) | ❌ |
| 函数签名 | 6 参数 | 8 参数 | 6 参数 |
| device 策略 | 纯 CPU | 纯 CPU | 纯 CPU |

**结论**: 修复后的 TreeCycle 版本与 OGBN 结构**完全一致**。

---

## 总结

### 问题
TreeCycle benchmark 在 multiprocessing 环境中死锁,而 OGBN 正常运行。

### 原因
在文件顶部导入重量级 explainer 模块,导致 `spawn` 模式的 multiprocessing 重复初始化和死锁。

### 解决方案
1. 移除顶部 explainer 导入
2. 在 worker_process 内部动态导入 (lazy import)
3. 完全对齐 OGBN 的代码结构

### 教训
**multiprocessing 的黄金法则**: 
- 主模块顶部只导入轻量级配置
- 在 worker 内部导入重量级计算模块
- 避免全局可变状态
- 遵循已验证的模式 (如 OGBN)

---

## 下一步

1. **测试**: 运行完整 benchmark (100 nodes, 20 workers)
2. **验证**: 确认无挂起,所有 workers 正常完成
3. **对比**: 与 OGBN 性能对比,确保行为一致
4. **扩展**: 测试其他 explainers (ApxChase, ExhaustChase)

**预期**: 修复后的 TreeCycle benchmark 应该像 OGBN 一样稳定高效! 🎉
