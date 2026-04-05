# backchase语义与coverage定义最终正确性检查

## 结论先行

当前主实验默认链路已经**基本满足**我们最终确认的 backchase 语义与 coverage 定义，可以开始正式实验，但还有两类残留需要明确知道：

1. 主实验默认链路已经统一到 `φ = (P, c)`、标准 backchase `c -> P`、local budget 按“每个 consequent 匹配实例一次 repair”解释。
2. `DBLP / Cora / MUTAG` 当前默认主链约束池都走 mined 路径，`c` 在主链上是 single-edge consequent，且带 target workload 上的 consequent 可匹配性筛选。
3. 同一个 workload 的 observed graph 在主实验 runner 中已经做到“一次生成、全方法复用”，并通过缓存键形成结构性保证；不是靠“同 seed 大概率一样”。
4. 还没有完全收干净的地方主要在主链之外：
   - 旧辅助脚本和旧数据集静态约束仍有历史残留
   - 它们不影响当前 `DBLP / Cora / MUTAG` 主实验默认路径，但不应再作为主链依据

一句话判断：

**当前默认主链已经可以作为正式实验链路使用。**

## 1. 术语与约束定义检查

### 1.1 旧两字段术语与错误旧拼写

全仓扫描结果：

- 未检出旧两字段术语残留
- 未检出错误旧拼写残留

扫描方式是精确匹配源码、配置、脚本和文档主目录；不包含输出目录与版本库内部目录。

### 1.2 当前约束是否统一成 `φ = (P, c)`

是。当前核心约束解释函数在：

- `src/grounding_semantics.py:48-59`

这里明确把约束解释成：

- `antecedent = P`
- `consequent = c`

`src/constraints.py:36-64` 的校验逻辑也只使用这两个字段。

## 2. 当前主链是否已经是标准 backchase `c -> P`

### 2.1 trigger 是否先匹配 `c`

是。关键位置：

- `src/grounding_semantics.py:123-133`
  - `_consequent_matches(...)` 只匹配 `c`
- `src/grounding_semantics.py:154-161`
  - `constraint_activation_summary(...)` 先取 `c` 的匹配
- `src/grounding_semantics.py:365-372`
  - `evaluate_grounding(...)` 先取 `c` 的匹配，再决定是否进入 repair
- `src/matcher.py:150-154`
  - `match_consequent_instances(...)` 也只匹配 `c`

当前主链里没有再把别的模式当作标准 trigger。

### 2.2 repair 是否是在补 `P`

是。关键位置：

- `src/grounding_semantics.py:376-396`
  - 对每个 `c` 的匹配实例，提取与 `P` 变量重合的绑定
  - 然后调用 `backchase_repair_cost(...)`
- `src/matcher.py:181-375`
  - `backchase_repair_cost(...)` 的输入就是 `antecedent_pattern`
  - repair 的搜索对象也是 `P` 的边集

所以当前真正运行中的语义是：

- 先匹配 `c`
- 再在 local budget 内补 `P`

## 3. 我们的方法与 baseline 的差异是否清楚

### 3.1 ApxC / HeuC / Exh

这三种方法都走：

- 先生成 witness `G_s`
- 再调用统一的 `evaluate_grounding(...)`
- 在其中执行 `G_s -> G_g` 的 backchase grounding

统一入口在：

- `src/apxchase.py:1235-1238`
- `src/heuchase.py:575-578`
- `src/exhaustchase.py:453-456`

三者的差异只保留在 candidate generation，不在 grounding 语义。

### 3.2 GEX / PGX

baseline 不做 `G_s -> G_g` 扩展。关键位置：

- `src/utils.py:605-677`

这里的 `compute_direct_constraint_coverage(...)` 明确说明：

- baseline 只评估单个 witness 自身是否严格满足约束
- 不构造 `G_g`
- 不允许借用 witness 外部边
- 不允许假设性新增 `ΔE`

因此：

- 我们的方法：`top-K witness set + G_s -> G_g`
- baseline：`single witness only + strict satisfaction only`

这条边界在当前代码里是清楚的。

## 4. `G_g` 是否仍然只是辅助结构

是。

关键证据：

- `src/grounding_semantics.py:312-430`
  - `evaluate_grounding(...)` 只在 witness 上附加 grounding metadata
  - 返回的是 grounded constraint 名称集合
- `src/grounding_semantics.py:257-300`
  - `attach_grounding_metadata(...)` 只是把 `delta_edges`、`supporting_edges`、`covered_constraints` 等信息挂回 witness 对象

没有任何位置把 repair 结果写回 observed graph `G`。

因此当前语义仍然是：

- `G_g` 只是辅助 grounded provenance graph
- observed graph 保持不变

## 5. 缺边可以补，缺点不可以补：当前是否落实

### 5.1 Active(Q) 的定义

当前代码已经落实成：

- `Hit_c(Q)`：`c` 可匹配
- `Active(Q)`：`c` 可匹配，且 `P` 的节点在当前 witness / workload 图里已经齐全

关键位置：

- `src/grounding_semantics.py:165-177`
  - 对每个 `c` 的匹配实例，调用 `_max_node_assignment(...)`
  - 只有当 `assigned_count == |V(P)|` 时，才把该约束记入 `Active(Q)`

这对应的是：

- 可以缺边
- 不能缺点

### 5.2 backchase repair 是否禁止借用 witness 外部节点

是。

关键位置：

- `src/grounding_semantics.py:344-345`
  - `witness_nodes` 和 `witness_edges` 从当前 witness 提取
- `src/grounding_semantics.py:388-395`
  - 调用 `backchase_repair_cost(...)` 时显式传入 `witness_nodes` 和 `witness_edges`
- `src/matcher.py:224-227`
  - 若提供了 `witness_nodes`，候选绑定节点只允许来自 `witness_nodes`
- `src/matcher.py:319-320`
  - 枚举现有边时也限制在 `witness_nodes` 内

所以当前 repair 只允许：

- 在当前 witness 的节点集上补边

不允许：

- 引入 witness 外部节点

## 6. `Hit_c(Q)` / `Active(Q)` / `Covered(Q)` 是否都已严格实现

是，当前三层集合都能在代码中找到明确落点。

### 6.1 `Hit_c(Q)`

来源：

- `src/grounding_semantics.py:154-164`
- `src/grounding_semantics.py:365-372`

只要 `c` 有匹配，就记入 `hit_names`。

### 6.2 `Active(Q)`

来源：

- `src/grounding_semantics.py:165-177`
- `src/grounding_semantics.py:376-381`

要求：

- `c` 有匹配
- 且 `P` 所需节点在当前图里已齐全

### 6.3 `Covered(Q)`

来源：

- `src/grounding_semantics.py:387-416`

逻辑是：

- 对每个 `c` 的匹配实例独立尝试 repair
- 只要存在一个实例能在 local budget 内补成 `P`
- 该约束就进入 `grounded_names`

这与我们确认的“exists one feasible repair”完全一致。

## 7. coverage 定义是否正确

### 7.1 workload 级别

当前 node 任务主路径：

- `src/Run_Experiment_Node.py:221-227`

当前 graph 任务主路径：

- `src/Run_Experiment.py` 中同样把 normalized coverage 作为单 workload 输出字段

当前 baseline：

- `src/utils.py:663-675`

都使用：

- `coverage_ratio_global(Q) = |Covered(Q)| / |Σ|`
- `coverage_ratio_normalized(Q) = |Covered(Q)| / |Active(Q)|`

且当 `|Active(Q)| = 0` 时：

- normalized coverage = `0`

### 7.2 数据集级别

汇总脚本在：

- `scripts/collect_results.py:223-227`

这里是：

- 先把每个 workload 的 `coverage_global` / `coverage_normalized` 写入 target-level CSV
- 再在 run-level对这些 workload 值取平均

不是把多 workload 的覆盖集合做 union 后当主 coverage。

所以当前主 coverage 口径是对的。

## 8. local budget `k` 是否真的是“每个 consequent 匹配实例一个”

是，这一点当前代码实现是正确的。

关键证据：

- `src/grounding_semantics.py:375-406`
  - `for bind_view in matches:` 逐个遍历 `c` 的匹配实例
  - 每个实例单独调用一次 `backchase_repair_cost(...)`
  - 每次调用都传入同一个 local budget `B`
- `src/matcher.py:265-375`
  - `backchase_repair_cost(...)` 内部的搜索和剪枝完全围绕单次 repair 展开
  - `if len(delta_edges) > B: return`

一条约束是否被 cover 的判定是：

- 只要存在一个 `c` 的匹配实例
- 该实例能在自己的 local budget `k` 内补成 `P`
- 该约束就被当前 workload cover

因此：

- 不是整条约束共享一个全局预算
- 不是整个 workload 共享一个全局预算

## 9. observed graph 是否在整条实验链中保持一致

### 9.1 node 任务

关键位置：

- `src/Run_Experiment_Node.py:120-158`

`_prepare_observed_node_workload(...)` 会：

- 按 `target_node + L + mask_ratio + seed + constraints_signature` 生成唯一 cache key
- 若缓存存在，直接复用
- 若缓存不存在，只生成一次 observed subgraph，并落盘缓存

然后五种方法都通过：

- `src/Run_Experiment_Node.py:497`
- `src/Run_Experiment_Node.py:537`
- `src/Run_Experiment_Node.py:575`
- `src/Run_Experiment_Node.py:588`
- `src/Run_Experiment_Node.py:601`

读取同一个 `observed_subgraph`。

### 9.2 graph 任务

关键位置：

- `src/Run_Experiment.py:82-126`

`_prepare_observed_graph_workload(...)` 也是同样的缓存模式。

五种方法都通过：

- `src/Run_Experiment.py:787`
- `src/Run_Experiment.py:798`
- `src/Run_Experiment.py:810`
- `src/Run_Experiment.py:849`
- `src/Run_Experiment.py:888`

复用同一个 `observed_graph`。

### 9.3 incompleteness factor study

当前 cache key 显式包含：

- `mask_ratio`

见：

- `src/Run_Experiment_Node.py:127-135`
- `src/Run_Experiment.py:92-100`

所以只有 incompleteness 作为前置实验因子变化时，observed graph 才会改变。

同时 masking 时间不计入方法时间：

- node 任务计时从 `src/Run_Experiment_Node.py:187` 开始
- graph 任务 baseline 计时从 `src/Run_Experiment.py:376` / `491` 开始
- 这些都发生在 observed graph 生成之后

因此当前 observed graph 一致性是结构性保证，不是“同 seed 大概率一样”。

## 10. 当前默认 `DBLP / Cora / MUTAG` 主链中，`c` 是否都是真实 single-edge consequent

### 10.1 DBLP

是。

关键位置：

- `src/constraint_mining.py:546-585`
  - DBLP mined 约束每条都只发出一个 `consequent_edge`
- `src/constraint_mining.py:588-633`
  - 还会按 target workload 上 `c` 的可匹配性筛选
- `configs/local/dblp.yaml`
  - `constraint_source: mined`
  - `constraint_filter_target_matchability: true`

### 10.2 Cora

默认主链是可以的。

关键位置：

- `src/constraint_mining.py:905-975`
  - `extension` 模板的 `c` 是单边
- `configs/local/cora.yaml`
  - 只启用 `extension`
  - 开启 `constraint_filter_target_matchability: true`

因此 `Cora` 默认主链里，`c` 是 single-edge consequent，且有可匹配性筛选。

### 10.3 MUTAG

默认主链也是可以的。

关键位置：

- `configs/local/mutag.yaml`
  - 只启用 `extension`
  - 开启 `constraint_filter_target_matchability: true`

因此默认主链里，`MUTAG` 的 `c` 也是 single-edge consequent。

### 10.4 仍然存在的非主链残留

`src/constraints.py` 里仍保留了一批旧静态约束定义，其中包括 multi-edge consequent 或不作为当前标准主链使用的内容，例如：

- `src/constraints.py:72-94`
- `src/constraints.py:536-546`

但它们当前**不在 `DBLP / Cora / MUTAG` 默认主实验链路中生效**。

## 11. 仅在 ApxC 上做的 local `k` probe

为了确认 coverage 偏低是不是主要由 local budget 卡住，我只在 `DBLP small regression` 上对 `ApxC` 做了一个轻量 probe。

结果：

| local k | avg_hit_constraint_count | avg_active_constraint_count | avg_covered_constraint_count | avg_coverage_global | avg_coverage_normalized | avg_conciseness | avg_fidelity_minus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15.6000 | 1.9333 | 1.8333 | 0.0917 | 0.3472 | 0.0187 | 0.0117 |
| 2 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.0187 | 0.0117 |
| 4 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.0187 | 0.0117 |
| 6 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.0187 | 0.0117 |

解释：

- `k=1 -> k=2` 有小幅提升
- `k=2 -> k=4/6` 没有继续提升

所以从代码定义和这个 probe 看：

- local budget 不是当前 coverage 偏低的首要瓶颈
- `k=2` 基本已经足够覆盖当前能通过 repair 的那部分实例

## 12. 如果当前 coverage 仍偏低，最可能受哪些因素影响

从当前实现定义上看，影响顺序更接近：

1. `Active(Q)` 太少  
   - `c` 虽然能命中，但满足 `P` 节点齐全的实例不多
2. consequent 可匹配但 node-complete 稀少  
   - 当前 `Hit_c(Q)` 明显大于 `Active(Q)`
3. candidate / witness 质量不足  
   - witness 没把足够多的相关节点和边带进来
4. local budget 偏小  
   - 但根据上面的 `ApxC` probe，这一项目前不是主因

## 13. 最终结论

当前默认主链已经满足下面这些关键要求：

- 约束统一成 `φ = (P, c)`
- 主算法是标准 backchase：`c -> P`
- 自有方法走 `G_s -> G_g`，baseline 不走 backchase
- `G_g` 不写回 observed graph
- `Hit_c(Q)` / `Active(Q)` / `Covered(Q)` 三层集合已实现
- coverage 双口径定义正确
- local budget 是“每个 consequent 匹配实例一个”
- observed graph 在主实验 runner 中一次生成、全链复用
- 默认 `DBLP / Cora / MUTAG` 主链约束池里的 `c` 都是 single-edge 且可匹配

因此：

**当前已经可以放心开始正式实验。**

保留意见只有一条：

- 仓库里仍有少量非主链旧静态约束定义和旧辅助脚本残留；它们不影响当前默认实验，但不应再拿来作为主链依据。
