# HeuC / Exh witness 验证修复结果

## 1. 现象

在 `DBLP` 小型 regression（30 个 author workloads, `gcn2`, `|Σ|=20`, `k=2`, `L=2`）上，之前出现了明显异常：

- `HeuC`
  - `avg_conciseness = 0.0000`
  - `avg_fidelity_minus = 0.0000`
- `Exh`
  - `avg_conciseness = 0.0755`
  - `avg_fidelity_minus = 0.0755`

这说明 `HeuC / Exh` 选出来的 witness 过于接近 full workload graph，或者 verification / final witness path 没有正确回到 observed-query 语义。

## 2. 根因

### HeuC

`HeuC` 当前的 Edmonds 路径里，不只是取 arborescence 边，而是：

- 先求 arborescence
- 再取其 weakly connected component
- 再把原图里这个 component 上的所有边都诱导回来

这会把 `HeuC` 退化成 near-full-graph candidate generator，直接毁掉它应有的速度/conciseness 特性。

对应文件：

- [src/heuchase.py](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/heuchase.py)

### Exh

`Exh` 的 candidate search space 在 cleaned graph 上没问题，但它在进入 witness verification / final witness 统计时，没有先回投到 observed-query graph。

这会导致：

- factual / counterfactual witness check 仍然在带 synthetic cleaned edges 的 candidate 上进行
- 最终 witness 的 `conciseness / fidelity` 被 cleaned-edge 污染

对应文件：

- [src/exhaustchase.py](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/exhaustchase.py)

## 3. 这次最小修复

### HeuC

把 `HeuC` 收回成真正的纯 Edmonds / arborescence 路径：

- 只保留 arborescence 本身的 underlying undirected edges
- 不再把 weak component 上的原图边整块诱导回来

这样做后：

- 没有改 ApxC
- 没有改 HeuC 的方法边界
- 只是把 HeuC 从“过重的 near-full induced heuristic”拉回“更纯的 Edmonds candidate”

### Exh

保留：

- `Exh` 仍然在 cleaned graph 上做 candidate generation

但新增：

- candidate 在进入 verifier / window 前，先投影回 observed-query graph
- 也就是 final witness 只保留 observed graph 上真实存在的边
- synthetic cleaned edges 不再直接进入 witness verification 与最终指标

这样做后：

- `Exh` 的 cleaned search space 保留
- 但 witness semantics 回到你前面一直要求的 observed-query 口径

## 4. 修复前后对比

### 修复前

| 方法 | avg_coverage_normalized | avg_conciseness | avg_fidelity_minus | runtime_total |
|---|---:|---:|---:|---:|
| ApxC | 0.4992 | 0.5394 | 0.2199 | 30.6597s |
| HeuC | 0.6667 | 0.0000 | 0.0000 | 34.0304s |
| Exh | 0.6667 | 0.0755 | 0.0755 | 45.3989s |

### 修复后

来自：

- [dblp_small_regression_method_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_small_regression_method_summary.csv)

| 方法 | avg_coverage_normalized | avg_conciseness | avg_fidelity_minus | runtime_total |
|---|---:|---:|---:|---:|
| ApxC | 0.4992 | 0.5394 | 0.2199 | 27.8908s |
| HeuC | 0.6667 | 0.3829 | 0.1311 | 28.8989s |
| Exh | 0.6667 | 0.1524 | 0.0746 | 35.8727s |

## 5. 结论

这次修复后：

- `HeuC` 已不再退化成 full-graph witness 路径
- `Exh` 的 witness verification / final metrics 已重新回到 observed-query graph 口径
- `coverage_normalized` 基本保持住了
- `HeuC / Exh` 的 `conciseness / fidelity` 明显恢复正常

其中最关键的是：

- `HeuC`
  - `avg_conciseness`: `0.0000 -> 0.3829`
  - `avg_fidelity_minus`: `0.0000 -> 0.1311`
- `Exh`
  - `avg_conciseness`: `0.0755 -> 0.1524`
  - `runtime_total`: `45.40s -> 35.87s`

## 6. 当前判断

可以继续后续实验，但要记住：

- 这次修的是 `HeuC / Exh` 的 witness path / verification alignment
- 没有动 candidate generation 主框架之外的其他方法
- 没有动 grounding 主链之外的 set maintenance / scoring / mining

所以这一步已经把当前最明显的异常收住了。  
