# constraint设计优化与coverage提升建议

## 1. 当前 coverage 偏低的主要结构原因

这一步只看当前标准 backchase 主链，不改算法、不改 candidate generation、不改 `UpdateWK`。

当前 `DBLP` 主链下，coverage 偏低的首要原因不是 `c (consequent)` 命中率不够，也不是 local budget `k=2` 太小，而是：

1. 很多规则虽然 `c` 能命中，但 `P (antecedent)` 对当前 workload 来说不够 node-complete。
2. 因而主要瓶颈出现在 `Hit_c(Q) -> Active(Q)` 这一层。
3. 一旦规则进入 `Active(Q)`，当前主链下通常就能被 cover，说明 repair cost 不是主因。

这点在当前 30 个 `DBLP` sampled workloads、默认 20 条规则池、`ApxC` 的基线结果里非常明显：

- `avg_hit_consequent_constraint_count = 15.6000`
- `avg_active_constraint_count = 1.9333`
- `avg_covered_constraint_count = 1.9333`
- `avg_coverage_global = 0.0967`
- `avg_coverage_normalized = 0.3667`

解释：

- `c` 平均一共能命中 `15.6` 条规则，说明 consequent 命中率已经不低。
- 但只有 `1.93` 条规则能进一步进入 `Active(Q)`，说明瓶颈在 `P` 的 node-complete 友好度。
- `Active(Q)` 与 `Covered(Q)` 基本相等，说明一旦 antecedent 所需节点齐全，当前 `k=2` 已经足够。

所以，**最该先改的是 constraint 设计与 rule ranking/filtering，不是先改 `k`，也不是先改主算法。**

## 2. consequent 设计建议

### 当前判断

当前 `DBLP` 默认主链里，`c` 都是 single-edge consequent，而且是 observed graph 中真实可匹配的关系。这一点现在已经满足标准 backchase 的要求。

从这次分析看，`c` 本身不是当前 coverage 的主要短板。

### 应保留的 `c`

优先保留下面这种 consequent：

- 单边关系
- 在 target workload 分布里稳定可见
- 距离 target 更近、局部性更强
- 命中后更容易回推出一个小而紧凑的 `P`

### 应降权或过滤的 `c`

即使某个 `c` 合法，也应降权或过滤：

- 在 target workloads 上很少命中
- 虽然命中，但几乎从不进入 `Active(Q)`
- 总是导向一个很大、很远、很难 node-complete 的 antecedent

### 结论

对当前 `DBLP` 主链来说，**consequent 设计是第二优先级**。  
第一优先级不是再找更高频的 `c`，而是找“命中后更容易回推出 `P`”的 `c`。

## 3. antecedent 设计建议

### 当前判断

当前 coverage 偏低最主要是因为很多 `P` 太大、太远、太碎，导致：

- `c` 已经命中
- 但 `P` 所需节点在当前 workload 里不齐
- 因而规则无法进入 `Active(Q)`

### 更适合当前标准 backchase 的 `P`

应优先 favor 下面这类 antecedent：

- 节点更少
- 边数更少，但仍有语义支撑
- 与 `c` 共享节点更多
- 更局部、更紧凑
- 从 `c` 回推时只差少量结构
- 不依赖长距离扩张或额外节点补入

### 这次 rule audit 的直接证据

当前 64 条候选规则池里，`active` 友好度最高的一批规则，大多属于：

- `dblp_apt_backchase`

这类规则的结构特征是：

- `P` 通常只有 `2` 个节点、`1` 条边
- `c` 也是 `1` 条边
- `P` 与 `c` 节点重叠高
- `Hit_c -> Active` 转化明显更好

相对地，很多当前默认池里的：

- `dblp_coauthor_topic_backchase`

虽然 consequent 命中很高，但：

- `P` 更大
- 节点更多
- 从 `c` 回推时更容易 node-incomplete

### 结论

对 coverage 提升来说，**antecedent 设计是第一优先级**。

## 4. filtering / ranking 建议

当前最站得住脚、同时又不改主算法的做法，不是重写 mining，而是在 mined 候选池之后做一层更合理的 ranking/filtering。

### 不建议继续只按 consequent support 排

如果只按 consequent support 排，会保留很多：

- `c` 很容易命中
- 但 `P` 几乎从不 active

的规则。

这类规则对当前标准 backchase 主链帮助不大。

### 更合理的 ranking 信号

对当前主链，更适合的 ranking 分数应综合：

1. consequent workload hit count
2. `Hit_c -> Active` 转化率
3. `Active -> Covered` 转化率
4. antecedent 节点数惩罚
5. antecedent 边数惩罚
6. `P` 与 `c` 的节点重叠奖励

### 最小 filtering 建议

先做一层轻量过滤，去掉这些规则：

- `c` 在 target workloads 上几乎总是 irrelevant
- `c` 命中后几乎从不 active
- `P` 太大且与 `c` 节点重叠太少
- `Active(Q)` 很低且几乎没有实际 coverage 贡献

## 5. 小型 before / after 对比

对比设置：

- dataset: `DBLP`
- method: `ApxC`
- workloads: 固定同一批 `30` 个 sampled workloads
- observed graph: 完全复用同一份缓存
- model: 相同
- `k = 2`
- `L = 2`
- 只改 constraint pool 的 filtering/ranking

对比文件：

- 原始规则审计：
  - [dblp_constraint_rule_audit.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_rule_audit.csv)
- 过滤后的 top-20 规则池：
  - [dblp_constraint_filtered_pool.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_filtered_pool.csv)
- 过滤后 ApxC per-workload：
  - [dblp_constraint_filtered_apx_per_workload.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_filtered_apx_per_workload.csv)
- before/after 汇总：
  - [dblp_constraint_before_after_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_before_after_summary.csv)

### before / after 汇总表

| 版本 | 规则数 | workloads | avg_hit_c | avg_active | avg_covered | avg_coverage_global | avg_coverage_normalized | nonzero_coverage_ratio | avg_conciseness | avg_fidelity_minus | runtime_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before_default_pool | 20 | 30 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.3667 | 0.0187 | 0.0117 | 17.0107 |
| after_filtered_pool | 20 | 30 | 12.7667 | 3.6667 | 3.6333 | 0.1817 | 0.5833 | 0.6000 | 0.3700 | 0.1828 | 15.4218 |

### 对比解读

过滤后规则池的表现是：

- `avg_hit_consequent_constraint_count` 下降  
  从 `15.6000` 降到 `12.7667`
- 但 `avg_active_constraint_count` 大幅上升  
  从 `1.9333` 升到 `3.6667`
- `avg_covered_constraint_count` 也大幅上升  
  从 `1.9333` 升到 `3.6333`
- `avg_coverage_global` 提升  
  从 `0.0967` 升到 `0.1817`
- `avg_coverage_normalized` 提升  
  从 `0.3667` 升到 `0.5833`
- `nonzero_coverage_workload_ratio` 提升  
  从 `0.3667` 升到 `0.6000`

这说明：

- 不是命中更多规则就更好
- 更重要的是命中的规则能不能进入 `Active(Q)`

也就是说，**更少但更“active-friendly”的规则池，比更多但 node-complete 友好度差的规则池更适合当前标准 backchase 主链。**

## 6. trade-off 与论文可接受性

这次 prototype 也有一个需要明确写出来的 trade-off：

- `avg_fidelity_minus` 从 `0.0117` 上升到 `0.1828`

按当前口径，`fidelity_minus` 越小越好。  
所以这说明：**只按 coverage 友好度做 filtering/ranking，虽然显著提升 coverage，但会带来 fidelity 代价。**

因此，这轮结果不能直接说明“过滤后规则池已经可以无条件替换默认池”，但它已经足够说明：

1. 当前 coverage 低的主因确实是规则结构问题。
2. 只通过 constraint 设计 / filtering / ranking，就能在不改主算法的前提下显著提升 coverage。
3. 下一步真正可落地的优化，不是重写主算法，而是在 rule ranking 里加入对 fidelity 的约束或惩罚。

## 7. 最小、最站得住脚的优化方案

在不改主算法的前提下，建议先做这套最小优化：

1. mined 阶段先挖更宽的候选池，例如 `64` 条。
2. consequent 仍保持：
   - single-edge
   - 真实可匹配
   - target workload 上可见
3. mined 后不再只按 consequent support 排序。
4. 主 ranking 改成优先看：
   - `Hit_c -> Active` 转化
   - antecedent 紧凑度
   - `P` 与 `c` 的节点重叠
5. 再把 fidelity 代价作为第二层约束，避免 coverage 上去了但 fidelity 明显变坏。

## 8. 结论：最该先改什么

如果目标是在保持当前标准 backchase 主链不变的前提下优先提升 coverage，那么最该先改的优先级是：

1. **antecedent 设计**
2. **rule ranking / filtering**
3. consequent 设计

更具体地说：

- 当前 coverage 偏低，首要不是 `c` 不命中。
- 首要是规则池里有太多“`c` 虽可命中，但 `P` 太难 active”的规则。
- 因此，最值得做的不是放宽预算，也不是改主算法，而是：
  - favor 更局部、更小、更紧凑的 antecedent
  - favor `Hit_c -> Active` 更高的规则
  - 去掉对当前 target workloads 几乎总是 irrelevant 的规则

## 9. 是否值得真正改 constraint pool

值得。

但第一步不应该是重写全套 mining 框架，而应该是：

- 保留当前标准 backchase 主链
- 在 mined 结果之后增加一层更合理的 ranking/filtering
- 先把规则池从“命中性优先”改成“active 友好度优先”

如果后续你要真正落到默认主链上，我建议下一步做：

1. 把这套 ranking/filtering 先只接到 `DBLP`
2. 再做同样 workload、同样 observed graph 的 `ApxC / HeuC / Exh` 对比
3. 同时监控：
   - `coverage_global`
   - `coverage_normalized`
   - `avg_fidelity_minus`
   - `avg_conciseness`

只有这四个指标一起看，才能决定是否值得把新的 constraint pool 变成默认实现。
