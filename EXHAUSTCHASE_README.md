# ExhaustChase Implementation

## 概述

ExhaustChase 是一个基于穷尽式 chase 的 baseline 方法，用于图解释生成。它与 ApxChase 的主要区别在于：

### 关键特性

1. **穷尽式规则修复（Exhaustive Enforcement）**
   - 在生成候选解释之前，先穷尽式地修复所有 TGD 规则违规
   - 持续迭代直到没有任何 TGD 违规存在（或达到最大迭代次数）
   - 在 TGD 的视角下，将"脏图"清理成"干净图"

2. **与 ApxChase 相同的候选生成**
   - 使用与 ApxChase 完全相同的方法生成候选
   - 维护相同的窗口机制
   - 计算相同的 grounding constraints

3. **性能开销体现**
   - 将穷尽式修复的时间开销包含在总时间统计中
   - 预期会比 ApxChase 慢得多，用于对比展示优化的价值

## 使用方法

### 配置文件设置

在 `config.yaml` 中：

```yaml
exp_name: exhaustchase_mutag
max_enforce_iterations: 100  # 最大修复迭代次数
```

### 运行实验

```bash
# 运行单个图
python src/Run_Experiment.py --input 0

# 运行所有测试图
python src/Run_Experiment.py --run_all
```

## 实验结果示例

在 MUTAG 数据集上的测试（graph_index=0）:

- **ApxChase**: 0.0552秒
- **ExhaustChase**: 7.9496秒
  - 修复阶段: 1.7534秒
  - 候选生成: 6.1962秒
  
ExhaustChase 慢了约 **144倍**，这展示了穷尽式 enforcement 的巨大开销。

## 实现细节

### 核心算法

```
ExhaustChase.explain_graph(G):
    1. H_clean, enforce_time ← exhaustive_enforce(G)
    2. Sigma*, S_k ← run_candidate_generation(H_clean)  # 与 ApxChase 相同
    3. return Sigma*, S_k, enforce_time
```

### 穷尽式修复流程

```
exhaustive_enforce(H):
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        violations_found = False
        
        for each TGD in Sigma:
            matches ← find_head_matches(H, TGD)
            for each binding in matches:
                feasible, cost, repairs ← backchase_repair_cost(H, TGD, binding, B)
                if not feasible or cost > 0:
                    violations_found = True
                    apply_repairs(H, repairs)  # 添加缺失的边
        
        if not violations_found:
            break  # 没有违规，修复完成
    
    return H_clean, enforce_time, iterations
```

## 输出信息

ExhaustChase 会输出详细的修复过程信息：

```
[ExhaustChase] 开始穷尽式规则修复...
[ExhaustChase] 迭代 10: 修复了 73 个违规, 图边数: 80
[ExhaustChase] 迭代 20: 修复了 73 个违规, 图边数: 110
...
[ExhaustChase] 修复阶段用时: 1.7534秒
[ExhaustChase] 清理后图: |V|=15, |E|=350
```

以及候选生成的时间分解：

```
[DEBUG] ExhaustChase total runtime: 7.9496s
[DEBUG]   - Enforcement overhead: 1.7534s
[DEBUG]   - Candidate generation: 6.1962s
```

## 参数说明

- `max_enforce_iterations`: 最大修复迭代次数（默认100）
  - 如果达到此限制，可能仍有违规存在
  - 可以根据图的复杂度调整

- 其他参数与 ApxChase 相同：
  - `L`: hop 数
  - `k`: 窗口大小
  - `B`: 修复预算
  - `alpha`, `beta`, `gamma`: 评分权重

## 局限性

1. **计算开销大**：穷尽式修复需要大量时间
2. **可能不收敛**：某些图可能需要超过 max_iterations 次迭代
3. **内存占用**：修复过程中会增加大量边，图可能变得很大

## 代码位置

- 实现：`src/exhaustchase.py`
- 实验入口：`src/Run_Experiment.py` 中的 `_run_one_graph_exhaustchase()`
- 配置：`config.yaml`
