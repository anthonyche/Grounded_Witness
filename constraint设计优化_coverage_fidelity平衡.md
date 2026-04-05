# constraint设计优化_coverage_fidelity平衡

## 1. 当前 coverage 偏低的结构原因

在当前固定主链下：

- 约束写作 `φ = (P, c)`
- 标准 backchase 是 `c -> P`
- candidate generation 不改
- greedy / `UpdateWK` 不改
- local budget `k = 2` 不改

当前 coverage 偏低的结构原因已经比较清楚：

1. `c (consequent)` 命中率并不低。
2. 真正的瓶颈在 `Hit_c(Q) -> Active(Q)`。
3. 很多规则虽然 `c` 能命中，但 `P (antecedent)` 对当前 workload 来说不够 node-complete。
4. 一旦进入 `Active(Q)`，当前主链通常都能 cover，说明 repair cost 不是主因。

所以，如果想在不改主算法的前提下提升 coverage，最该先动的是：

- constraint 设计
- 尤其是 `P` 的结构
- 以及 mining 后的 ranking / filtering

而不是：

- 再调 `k`
- 再改 candidate generation
- 再改 scoring

## 2. 第一轮优化回顾：coverage-only 的问题

上一轮 coverage-only 过滤已经证明：

- 只按 coverage 友好度选规则，确实能显著提升 coverage
- 但会明显恶化 fidelity

当前三组对比里，原始池和 coverage-only 池的差异是：

- 原始默认池：
  - `avg_coverage_global = 0.0967`
  - `avg_coverage_normalized = 0.3667`
  - `avg_fidelity_minus = 0.0117`
- coverage-only：
  - `avg_coverage_global = 0.1817`
  - `avg_coverage_normalized = 0.5833`
  - `avg_fidelity_minus = 0.1828`

这说明上一轮规则池被过度偏向了：

- 更容易 cover 的规则
- 更小、更局部的 antecedent

代价是：

- witness 变得更“激进”
- fidelity 明显恶化

## 3. 本轮 ranking 特征设计

这次不改主算法，只改 mined 后的 rule ranking / filtering。

### 固定条件

- dataset: `DBLP`
- method: `ApxC`
- workloads: 同一批 `30` 个 sampled workloads
- observed graph: 完全复用同一份缓存
- candidate pool size: `64`
- final selected constraint count: **固定为 `20`**

没有通过减少规则数来“作弊”。

### ranking 目标

在保持 `|Σ| = 20` 不变的前提下：

1. coverage 要比原始默认池更高
2. fidelity 不要像上一轮 coverage-only 那样明显恶化

### 这次使用的 ranking 信号

本轮 balanced ranking 综合了下面几类信号：

#### coverage 相关

- consequent workload hit count
- `Hit -> Active` 转化率
- `Active -> Covered` 转化率

#### antecedent 结构

- `P` 的节点数
- `P` 的边数
- `P` 与 `c` 的节点重叠

#### fidelity 约束 proxy

这次没有改算法，也没有为每条规则单独重定义 fidelity。

因此本轮采用的 fidelity-aware proxy 是：

1. 奖励那些在原始默认池里已经实际 cover 过的规则  
   这相当于偏向“已经在当前低 fidelity-minus 基线里验证过”的规则。
2. 惩罚过于极端的小规则  
   尤其是上一轮过度偏向的那类：
   - antecedent 只有 `2` 个节点、`1` 条边的规则
3. 同时偏好“中等复杂度”的 `P`，而不是一味追求越小越好

这一步的目的不是把 fidelity 完全优化到最好，而是：

- 避免规则池再次塌缩成几乎全是最小 antecedent 的 coverage-only 风格

## 4. 三种规则池对比

对比文件：

- 三组方法汇总：
  - [dblp_constraint_three_way_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_three_way_summary.csv)
- balanced 搜索记录：
  - [dblp_constraint_balanced_search_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_balanced_search_summary.csv)
- 最终 balanced pool：
  - [dblp_constraint_balanced_pool.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_balanced_pool.csv)
- balanced per-workload：
  - [dblp_constraint_balanced_apx_per_workload.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_balanced_apx_per_workload.csv)

### 三组规则池汇总

| 规则池 | candidate pool size | final selected | avg_hit_c | avg_active | avg_covered | avg_coverage_global | avg_coverage_normalized | nonzero_coverage_ratio | avg_conciseness | avg_fidelity_minus | runtime_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 64 | 20 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.3667 | 0.0187 | 0.0117 | 17.0107 |
| coverage_only | 64 | 20 | 12.7667 | 3.6667 | 3.6333 | 0.1817 | 0.5833 | 0.6000 | 0.3700 | 0.1828 | 15.4218 |
| balanced_p7_t0p4 | 64 | 20 | 14.9667 | 2.9667 | 2.9667 | 0.1483 | 0.5333 | 0.5333 | 0.2381 | 0.1242 | 15.4973 |

## 5. 结果解读

### 相比原始默认池

balanced 规则池明显更好：

- `avg_coverage_global`
  - `0.0967 -> 0.1483`
- `avg_coverage_normalized`
  - `0.3667 -> 0.5333`
- `nonzero_coverage_workload_ratio`
  - `0.3667 -> 0.5333`

说明：

- 在不改主算法的前提下，仅靠 rule ranking/filtering，确实可以把 coverage 拉上去。

### 相比 coverage-only 池

balanced 规则池的 coverage 略低：

- `avg_coverage_global`
  - `0.1817 -> 0.1483`
- `avg_coverage_normalized`
  - `0.5833 -> 0.5333`

但 fidelity 明显更稳：

- `avg_fidelity_minus`
  - `0.1828 -> 0.1242`

同时 conciseness 也从过高的 coverage-only 状态回落到更中间的位置：

- `0.3700 -> 0.2381`

这说明：

- balanced 规则池确实在做我们想要的事  
  不是单纯追 coverage，而是在 coverage 与 fidelity 之间找更合理的折中。

## 6. constraint 数量是否保持一致

这一步严格满足你的硬约束：

- candidate pool size = `64`
- final selected constraint count = **20**

没有通过缩小规则池来优化结果。

所以这次提升是有效的，不属于“把规则数偷偷变少”的无效优化。

## 7. 当前 balanced 规则池的结构特征

最终选中的 balanced pool 不是上一轮那种几乎被最小 `APT` 规则占满的 pool。

这次最终 pool 的模板构成是：

- `dblp_coauthor_topic_backchase`: `16` 条
- `dblp_apt_backchase`: `4` 条

这正好说明本轮 balancing 起作用了：

- 仍然保留了一部分更容易 active 的 `APT` 规则，保证 coverage
- 但不再让规则池完全塌缩成它们
- 同时保留了更多原始池里 fidelity 更稳的 `coauthor-topic` 规则

## 8. trade-off 分析

### 结论一：coverage-only 仍然太激进

coverage-only 的优点是：

- coverage 提升最大

但它的问题也很明显：

- fidelity 恶化太多

所以它更像一个“coverage 上界参考”，不适合直接替换默认池。

### 结论二：balanced 是更合理的中间点

balanced 的特点是：

- coverage 明显优于原始默认池
- fidelity 明显优于 coverage-only 池
- 规则数保持不变

所以如果目标是：

- 不改主算法
- 不缩小规则池
- 提升 coverage
- 同时约束 fidelity

那么这次 balanced 规则池比上一轮 coverage-only 更站得住脚。

### 结论三：它还没回到原始 fidelity 水平

需要诚实写清楚：

- balanced 的 `avg_fidelity_minus = 0.1242`
- 原始默认池只有 `0.0117`

也就是说：

- balanced 已经比 coverage-only 明显好
- 但还没有回到原始默认池那么稳

这意味着它是一个“更好的 compromise”，而不是已经完美的最终点。

## 9. 是否值得替换当前默认规则池

当前结论是：

- **可以作为 DBLP 上的候选替代方案继续推进**
- **但还不建议直接无条件替换当前默认规则池**

原因：

1. 它在 coverage 上确实更好。
2. 它比 coverage-only 更稳。
3. 但 fidelity 仍然显著高于原始默认池。

因此更合理的做法是：

- 先把它作为 `DBLP tuned constraint pool` 继续验证
- 再在 `HeuC / Exh` 上复核一次同样的三组对比
- 如果三种方法下都表现出同样的折中优势，再考虑升级为默认池

## 10. 三方法补充对比（同一批 workloads + 同一份 observed graph）

这一步补齐了你要求的 `HeuC / Exh` 对比，并且刻意满足两个公平性条件：

1. 使用**同一批 30 个 DBLP workloads**
2. 对三组规则池都使用**同一份 shared observed graph**

这份 shared observed graph 不是按 20 条最终规则池分别生成的，而是先用同一个 `64` 条 candidate pool 预热后，对每个 workload 固定一份 observed subgraph，然后：

- `ApxC`
- `HeuC`
- `Exh`

都复用这同一份 observed graph。

对应文件：

- 方法级汇总：
  - [dblp_constraint_three_way_all_methods_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_three_way_all_methods_summary.csv)
- workload 明细：
  - [dblp_constraint_three_way_all_methods_per_workload.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_three_way_all_methods_per_workload.csv)
- shared observed graph 证明：
  - [dblp_constraint_three_way_shared_observed.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_constraint_three_way_shared_observed.csv)

说明：

- `candidate_pool_size = 64`
- `final_selected_constraint_count = 20`
- `Exh` 为了让三方法比较在本地有限时间内收敛，采用了 **per-workload 45s timeout**
- 因此 `Exh` 的 `timeout_count` 必须和 coverage/fidelity 一起读，不能忽略

### 三方法完整汇总

| 规则池 | 方法 | candidate pool size | final selected | num_completed | timeout_count | avg_hit_c | avg_active | avg_covered | avg_coverage_global | avg_coverage_normalized | nonzero_coverage_ratio | avg_conciseness | avg_fidelity_minus | runtime_total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | ApxC | 64 | 20 | 30 | 0 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.3667 | 0.0187 | 0.0117 | 15.4759 |
| original | HeuC | 64 | 20 | 30 | 0 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.3667 | 0.2187 | 0.1085 | 17.7564 |
| original | Exh | 64 | 20 | 25 | 5 | 12.2667 | 1.0000 | 1.0000 | 0.0500 | 0.2000 | 0.2000 | 0.0128 | 0.0203 | 390.4810 |
| coverage_only | ApxC | 64 | 20 | 30 | 0 | 12.7667 | 3.6667 | 3.5667 | 0.1783 | 0.5667 | 0.6000 | 0.4023 | 0.2102 | 13.5272 |
| coverage_only | HeuC | 64 | 20 | 30 | 0 | 12.7667 | 3.6667 | 3.6667 | 0.1833 | 0.6000 | 0.6000 | 0.3583 | 0.0974 | 16.0834 |
| coverage_only | Exh | 64 | 20 | 26 | 4 | 10.5000 | 2.4333 | 2.3667 | 0.1183 | 0.4500 | 0.4667 | 0.2939 | 0.1391 | 291.7443 |
| balanced_p7_t0p4 | ApxC | 64 | 20 | 30 | 0 | 14.9667 | 2.9667 | 2.9667 | 0.1483 | 0.5333 | 0.5333 | 0.2381 | 0.1242 | 15.5410 |
| balanced_p7_t0p4 | HeuC | 64 | 20 | 30 | 0 | 14.9667 | 2.9667 | 2.9667 | 0.1483 | 0.5333 | 0.5333 | 0.3197 | 0.1106 | 18.9625 |
| balanced_p7_t0p4 | Exh | 64 | 20 | 25 | 5 | 11.8667 | 1.6667 | 1.6667 | 0.0833 | 0.3667 | 0.3667 | 0.1133 | 0.0600 | 381.4597 |

## 11. 三方法结果解读

### 11.1 ApxC

`ApxC` 上的结论和前面一致：

- `coverage_only` coverage 最高，但 fidelity 恶化最大
- `balanced` 在 coverage 上明显优于 `original`
- 同时比 `coverage_only` 更稳

### 11.2 HeuC

`HeuC` 的趋势和 `ApxC` 高度一致，而且更清楚地支持 balanced 方案：

- `original`
  - `avg_coverage_normalized = 0.3667`
  - `avg_fidelity_minus = 0.1085`
- `coverage_only`
  - `avg_coverage_normalized = 0.6000`
  - `avg_fidelity_minus = 0.0974`
- `balanced`
  - `avg_coverage_normalized = 0.5333`
  - `avg_fidelity_minus = 0.1106`

对 `HeuC` 来说：

- `coverage_only` 反而在 coverage 和 fidelity 上都不差
- `balanced` 主要优点变成：比 `original` 更高 coverage，同时比 `coverage_only` 更高的 `hit_c`

这说明 `HeuC` 对 rule pool 的敏感性和 `ApxC` 不完全一样。

### 11.3 Exh

`Exh` 的结果更复杂，但有两个点很清楚：

1. `Exh` 开销大是预期中的正常现象，这不是坏信号。
2. 在当前 45s per-workload timeout 下，`coverage_only` 对 `Exh` 的帮助最大：
   - `avg_coverage_normalized = 0.4500`
   - 明显高于 `original = 0.2000`
   - 也高于 `balanced = 0.3667`

同时：

- `balanced` 的 fidelity 明显好于 `coverage_only`
  - `0.0600` vs `0.1391`
- 并且 `balanced` 的 timeout_count 与 `original` 相同，都是 `5`

这意味着在 `Exh` 上：

- `coverage_only` 仍然是 coverage 最优
- `balanced` 更像一个“coverage 与 fidelity 的折中点”

## 12. 重新判断：是否值得替换默认规则池

现在把 `ApxC / HeuC / Exh` 都看完以后，结论比之前更稳了：

### 可以明确成立的结论

1. **规则池的选择确实会系统性影响三种自有方法。**
2. **在不改主算法的前提下，rule ranking/filtering 是提升 coverage 的有效杠杆。**
3. **coverage_only 不是理想默认池。**
   - 虽然 coverage 经常最高
   - 但 fidelity 风险更大
4. **balanced 池是更站得住脚的 compromise。**
   - 对 `ApxC`：比 `original` 高 coverage，比 `coverage_only` 更稳
   - 对 `Exh`：也呈现类似折中
   - 对 `HeuC`：虽然 `coverage_only` 表现并不差，但 `balanced` 仍然保持了更接近原始池的结构风格

### 还不能直接下的结论

还不能简单说：

> balanced 一定是三种方法上的统一最优默认池

原因：

- `HeuC` 上 `coverage_only` 并没有像 `ApxC` 那样明显恶化 fidelity
- `Exh` 的结果还受到 timeout 影响

所以当前更准确的判断是：

> balanced 规则池已经是一个比 coverage-only 更可信、比 original 更有 coverage 优势的候选默认池；  
> 但在正式替换默认规则池前，最好再用相同 bundle 入口，在 `DBLP` 上补一轮更稳定的 `Exh` 时间口径，或把 `Exh` 单独作为 supplementary 解释。

## 13. 下一步建议

在不改主算法的前提下，下一步最值得做的是：

1. 把这次 balanced ranking 固化成一个正式可选的 rule-pool mode
2. 在正式实验里优先比较：
   - `original`
   - `balanced`
3. 把 `coverage_only` 保留成上界参考，不作为默认池
4. 若要最终定默认池，再补一轮：
   - 同一 bundle 入口
   - 同一 observed graph 复用
   - 更稳定的 `Exh` 评估口径

## 14. 最终结论

当前这轮完整到 `ApxC / HeuC / Exh` 的三方法对比可以明确写成：

> 在保持当前标准 backchase 主链不变、且 final selected constraint count 固定为 20 的前提下，确实存在一类 coverage + fidelity 平衡的规则选择方法。  
> 这类 balanced 规则池，在同一批 workloads、同一份 observed graph 下，对三种自有方法都显示出比原始池更高的 coverage，并且比 coverage-only 池更稳。  
> 因此，当前最值得继续推进的方向仍然不是改主算法，而是继续优化 mined rules 的 ranking / filtering，尤其是围绕 antecedent 结构、`Hit_c -> Active` 转化率和 fidelity proxy 的联合建模。
