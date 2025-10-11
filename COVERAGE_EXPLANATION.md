# Coverage指标说明

## 概述

Coverage指标用于衡量生成的解释（witness/explanation）覆盖了多少个约束条件（constraints）。

## 计算公式

```
Coverage = |covered_constraints| / |total_constraints|
```

其中：
- `|covered_constraints|`: 被解释覆盖的约束数量
- `|total_constraints|`: 总约束数量（对于MUTAG数据集，共有8个TGD约束）

## Matching逻辑

为了判断一个约束是否被覆盖，我们使用与ApxChase相同的constraint matching策略：

### 1. Head Pattern Matching
对于每个constraint，在解释子图上寻找head pattern的匹配：
```python
head_matches = find_head_matches(subgraph, constraint)
```

### 2. Repair Cost Validation
对于找到的每个head match，计算backchase repair cost：
```python
cost = backchase_repair_cost(subgraph, constraint, match, Budget)
```

### 3. Coverage判定
如果满足以下条件，则该constraint被覆盖：
- `head_matches`非空（至少有一个head pattern match）
- 至少有一个match的repair cost ≤ Budget（默认Budget=8）

```python
if cost is not None and cost <= Budget:
    covered_constraints.add(constraint_name)
```

## 各方法的Coverage计算

### 对于Chase-based方法（ApxChase, ExhaustChase, HeuChase）

这些方法在生成witness时已经使用了TGD约束，因此：
1. 在`explain_graph()`执行过程中，已经收集了`Sigma_star`（被触发的约束集合）
2. Coverage直接基于`Sigma_star`计算：
   ```python
   coverage_names = [c.get("name") for c in Sigma_star]
   coverage_ratio = len(coverage_names) / len(constraints)
   ```

### 对于GNN Explainer方法（GNNExplainer, PGExplainer）

这些方法生成edge_mask但不直接使用constraints，因此：
1. 从edge_mask中选择top-k个边构建解释子图
2. 在该子图上应用constraint matching逻辑：
   ```python
   # 构建top-k边的子图
   k = config.get("gnnexplainer_topk", 10)
   topk_indices = torch.topk(edge_mask_flat, min(k, len(edge_mask_flat))).indices
   subgraph = Data(x=graph.x, edge_index=graph.edge_index[:, topk_indices])
   
   # 计算coverage
   covered_constraints, coverage_ratio = compute_constraint_coverage(
       subgraph, constraints, Budget=8
   )
   ```

## 代码实现

### utils.py - compute_constraint_coverage()

```python
def compute_constraint_coverage(subgraph, constraints, Budget=8):
    """计算解释子图覆盖了多少个约束"""
    from matcher import find_head_matches, backchase_repair_cost
    
    covered_names = []
    
    for constraint in constraints:
        constraint_name = constraint.get("name", str(constraint))
        head_matches = find_head_matches(subgraph, constraint)
        
        if not head_matches:
            continue
        
        for match in head_matches:
            cost = backchase_repair_cost(subgraph, constraint, match, Budget)
            if cost is not None and cost <= Budget:
                covered_names.append(constraint_name)
                break  # 一个constraint只需要一个有效match即可
    
    covered_names = sorted(set(covered_names))
    coverage_ratio = len(covered_names) / len(constraints)
    
    return covered_names, coverage_ratio
```

### Run_Experiment.py - 集成

所有方法的`_run_one_graph_*`函数都返回5个值：
```python
return elapsed, count, avg_fidelity, avg_conciseness, coverage_ratio
```

在`main()`函数中收集coverage_scores：
```python
coverage_scores: List[float] = []
...
elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_xxx(...)
coverage_scores.append(cov)
```

最后输出Coverage Statistics：
```python
overall_avg_coverage = float(np.mean(coverage_scores))
print(f"Overall Average Coverage: {overall_avg_coverage:.6f} ({overall_avg_coverage*100:.2f}%)")
```

## 结果示例

### ApxChase (图41)
```
Covered constraints (6/8 = 75.00%): 
['amine_di_carbon_completion', 'c6_closure', 'c6_closure_2', 
 'c6_closure_3', 'halogen_anchor_completion', 'nitro_on_aromatic_completion']

Coverage Statistics:
Overall Average Coverage: 0.750000 (75.00%)
```

### GNNExplainer (图39, top-10边)
```
Covered constraints (0/8 = 0.00%): []

Coverage Statistics:
Overall Average Coverage: 0.000000 (0.00%)
```

## JSON文件格式

每个方法的metrics_graph_*.json都包含以下coverage字段：

```json
{
  "coverage_size": 6,
  "covered_constraints": [
    "amine_di_carbon_completion",
    "c6_closure",
    "c6_closure_2",
    "c6_closure_3",
    "halogen_anchor_completion",
    "nitro_on_aromatic_completion"
  ],
  "total_constraints": 8,
  "coverage_ratio": 0.75
}
```

## 参数配置

### Budget
- 默认值：8
- 在config.yaml中配置：`Budget: 8`
- 影响：更高的Budget允许更宽松的constraint matching

### topk（仅用于GNN Explainer方法）
- GNNExplainer: `gnnexplainer_topk: 10`
- PGExplainer: `pgexplainer_topk: 10`
- 影响：选择更多的边可能提高coverage，但会降低conciseness

## 分析脚本

使用`analyze_fidelity.py`对比所有方法：

```bash
python analyze_fidelity.py
```

输出包括：
- 各方法的Overall Avg Coverage
- 按Coverage降序排名
- Coverage的min/max/std统计

## 意义

Coverage指标揭示了：
1. **Constraint-aware vs. Constraint-agnostic**: Chase-based方法（使用TGD约束）通常有更高的coverage，而纯数据驱动方法（GNN explainers）可能coverage较低
2. **解释的完整性**: 高coverage表示解释覆盖了更多的domain knowledge（约束条件）
3. **方法对比**: 结合Fidelity-和Conciseness，coverage提供了第三个维度来评估解释质量

## 总结

Coverage = 75%意味着：
- 在8个MUTAG约束中，解释覆盖了6个
- 解释子图能够满足这6个约束的head pattern并且repair cost在预算内
- 这些约束代表了分子结构中的重要模式（如苯环、氨基、硝基等）
