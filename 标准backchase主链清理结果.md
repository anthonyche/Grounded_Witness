# 标准 backchase 主链清理结果

## 1. 结论摘要

当前项目的**主实验链路**已经基本收敛到唯一正确版本：

- 约束统一解释为 `φ = (P, c)`
- `P = antecedent`
- `c = consequent`
- 标准 backchase 统一为 `c -> P`
- 我们的方法只在 `G_s -> G_g` 的 grounding 过程中执行 backchase
- baseline 不执行 `G_s -> G_g` 扩展，只做单个 witness 的严格满足判定
- 主实验 runner 中，同一个 workload 的 observed graph 已经做到**一次生成、全链复用**

另外，`Exh` 开销显著更高在当前语义下是**预期行为**，不作为主链不一致的证据。它本身就是更重的 clean-first baseline，较高成本可以与较强约束质量同时成立。

但如果把范围扩大到**整个仓库的所有遗留静态约束与辅助脚本**，项目还**没有完全达到“全仓唯一主链”**。当前还剩两类不一致：

1. 非 DBLP 的遗留静态约束池中，仍有一批 consequent 不是单边 `c`，如果手动启用，会偏离当前标准 backchase 主链。
2. 若使用主实验 runner 之外的辅助脚本，仍可能各自独立生成 observed graph；主实验链已做结构性保证，但全仓所有工具脚本尚未统一收口。

因此，当前最准确的判断是：

- **主实验默认链路：可以作为后续实验唯一主链**
- **整个仓库所有历史路径：还没有完全清空到只剩一个版本**

---

## 2. 旧术语与旧逻辑清理情况

### 2.1 旧两字段术语与错误旧拼写

对仓库执行全文扫描：

- 扫描范围：项目目录，排除 `.git`
- 结果：在当前工作区文件中，旧两字段术语与错误旧拼写的匹配结果为空

这说明当前活跃代码、注释、文档与报告中，已经不再残留这些旧术语。

### 2.2 当前是否已经统一为 `P / c`

是。当前主链相关代码已经统一使用：

- `antecedent`
- `consequent`

关键证据：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:48`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:48)
  - `interpret_constraint(...)` 只返回 `antecedent` 与 `consequent`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:150`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:150)
  - `match_consequent_instances(...)` 直接读取 `constraint["consequent"]`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:181`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:181)
  - `backchase_repair_cost(...)` 直接接受 `antecedent_pattern`

---

## 3. 当前主链是否已经只剩标准 `c -> P`

### 3.1 当前主链答案

主实验默认链路里，答案是：**是**。

当前真正运行中的 grounding / backchase 主链已经收敛为：

1. 先匹配 consequent `c`
2. 再在 budget 内补 antecedent `P`

关键证据如下。

### 3.2 consequent 匹配

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:123`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:123)
  - `_consequent_matches(...)` 只对 `consequent_pattern` 做匹配
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:153`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:153)
  - `constraint_activation_summary(...)` 先执行 consequent 匹配，再决定 hit / active

### 3.3 antecedent 修复

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:181`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:181)
  - `backchase_repair_cost(...)` 的唯一目标是补足 antecedent `P`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:190`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:190)
  - 注释已经明确：给定 consequent 匹配绑定后，估计满足 antecedent `P` 所需的最小缺边代价

### 3.4 已废弃的旧主触发逻辑

当前主实验相关实现中，已经看不到以下逻辑作为主触发对象：

- 以某个“完整目标模式”作为主触发目标
- 以“最大可观测匹配”作为标准 backchase 主触发

在当前活跃主链里，activate / active 的判定已经收口到：

- `c` 是否可匹配
- `P` 所需节点是否齐全

所以主实验默认路径的触发语义，已经是标准 `c -> P`。

---

## 4. 我们的方法与 baseline 的差异是否已经说清楚

### 4.1 我们的方法

`ApxC / HeuC / Exh` 现在共享同一套 grounding 语义：

- 输入：witness `G_s`
- 过程：`G_s -> G_g`
- 语义：在 budget 内做 backchase，把 antecedent `P` 补足
- 结果：`G_g` 只是辅助 grounded provenance graph

这三种方法的差异只保留在 candidate generation：

- `ApxC`：ranked bounded candidate generation
- `HeuC`：heuristic / Edmonds candidate generation
- `Exh`：cleaned-graph candidate generation

grounding / backchase 语义本身已经统一。

### 4.2 baseline

baseline 不走 `G_s -> G_g`。

关键证据：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/utils.py:605`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/utils.py:605)
  - `compute_direct_constraint_coverage(...)`
  - 文档与实现都明确说明：
    - baseline 不构造 grounded provenance graph
    - baseline 不执行 backchase 扩展
    - 只有单个 witness 自身严格满足约束时，才记为覆盖

因此这层差异已经清楚：

- 我们的方法：`top-K witness set` + `G_s -> G_g` backchase grounding
- baseline：`single witness only` + `strict satisfaction only` + `no backchase grounding`

---

## 5. observed graph 是否已经做到一次生成、全链复用

### 5.1 主实验 runner：已经做到

在主实验 graph runner 中：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:82`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:82)
  - `_prepare_observed_graph_workload(...)`
  - 对每个 graph workload 生成并缓存唯一 observed graph
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:103`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:103)
  - 若缓存存在，直接复用，不重新 masking
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:785`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:785)
  - `ApxC`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:796`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:796)
  - `GEX`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:807`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:807)
  - `PGX`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:819`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:819)
  - `Exh`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:858`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:858)
  - `HeuC`

这些方法都复用同一个 `_prepare_observed_graph_workload(...)` 输出。

在主实验 node runner 中：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:120`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:120)
  - `_prepare_observed_node_workload(...)`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:137`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:137)
  - 若缓存存在，直接复用
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:495`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:495)
  - `ApxC`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:507`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:507)
  - `HeuC`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:547`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:547)
  - `Exh`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:585`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:585)
  - `GEX`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:598`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:598)
  - `PGX`

所以：

- **主实验链路中，同一个 workload 的 observed graph 只生成一次**
- **然后被所有方法复用**

### 5.2 masking 时间是否计入方法时间

当前主实验 runner 中：

- 先调用 observed graph 准备函数
- 后进入各方法的 `t0 = time.time()` 到 `t1 = time.time()`

例如：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:787`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:787)
  observed graph 先生成
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:788`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:788)
  才进入方法运行

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:497`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:497)
  observed graph 先生成
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:498`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:498)
  才进入方法运行

因此，对主实验链而言：

- incompleteness 导致的 graph construction / masking 是前置步骤
- **不计入方法运行时间**

### 5.3 还剩的非统一点

主实验 runner 已经做到了结构性复用，但仓库里仍有一些**辅助脚本**会各自单独生成 observed graph，例如：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/visualize_case_study.py:482`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/visualize_case_study.py:482)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_cora_sensitivity.py:104`](/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_cora_sensitivity.py:104)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_cora_tuning.py:113`](/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_cora_tuning.py:113)

这些不影响当前主实验默认链，但说明“全仓所有工具脚本”还没有完全统一到唯一 observed graph 入口。

---

## 6. constraint mining 是否已经保证 `c` 真实可匹配、非虚拟

### 6.1 DBLP 默认 mined 路径

对 DBLP 默认 mined 路径，答案基本是：**是**。

关键证据：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:581`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:581)
  - `_filter_dblp_constraints_by_consequent_matchability(...)`
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:601`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:601)
  - 直接在 target workload 样本上用 `find_pattern_matches(sample, tgd["consequent"])`
    检查 consequent 可匹配性
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:623`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraint_mining.py:623)
  - 若启用 `constraint_filter_target_matchability`，则仅保留可匹配的约束

这意味着：

- DBLP 默认 mined constraint pool 已经显式筛掉了在目标 workload 上 consequent 基本不可匹配的约束
- 当前主链不再依赖虚拟 consequent

### 6.2 DBLP 静态约束

DBLP 静态约束已经被禁用：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraints.py:689`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraints.py:689)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraints.py:692`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/constraints.py:692)

当前状态是：

- `CONSTRAINTS_DBLP = []`

所以 DBLP 旧静态池已经不再污染主实验链。

### 6.3 仍残留的仓库级风险

虽然 DBLP 默认主链已经满足要求，但全仓库静态约束池里仍有一批 consequent 不是单边 `c`。

本次核查结果：

- 静态约束集合数：`7`
- consequent 不是单边的遗留约束数：`27`

代表性例子包括：

- `CONSTRAINTS_MUTAG / c6_closure`
- 多条 `CONSTRAINTS_CORA / cora_*`
- `CONSTRAINTS_BASHAPE / bashape_top_middle_bottom_closure`
- 多条 `CONSTRAINTS_OGBN_PAPERS / ogbn_*`
- 多条 `CONSTRAINTS_TREECYCLE / treecycle_*`
- 多条 `CONSTRAINTS_YELP / yelp_*`

这说明：

- **主实验默认链**已经基本符合 `(P, c)` 且单边 consequent 的要求
- **整个仓库的全部历史静态池**还没有完全收敛成这一形式

---

## 7. active / coverage 是否与标准 `c -> P` 一致

当前主实验链中，这一部分已经一致。

关键证据：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:136`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/grounding_semantics.py:136)
  - `constraint_activation_summary(...)`
  - 先 consequent 匹配，再根据 antecedent 节点齐全性判断 active

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:322`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:322)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:222`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:222)
  - hit / active 都来自同一套 activation summary

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:325`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:325)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:225`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:225)
  - 同时输出：
    - global coverage
    - normalized coverage

因此，当前主链的评估输出已经与 `c -> P` 的标准路径对齐。

---

## 8. `G_g` 是否仍然只是辅助 grounded provenance graph

当前实现仍然满足这条要求。

关键证据：

- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:194`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:194)
  - supporting edge 与 grounded edge 的区分仍然存在
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:195`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:195)
- [`/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:196`](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/matcher.py:196)
  - supporting edge 在 observed graph 中存在但不在 `G_s`
  - grounded edge 不在 observed graph 中并计入 `ΔE`

当前没有看到把 grounded provenance 边写回 observed graph `G` 的实现。  
主链上仍然是：

- observed graph 固定
- `G_s -> G_g` 只是解释与统计中的辅助扩展

---

## 9. 还剩哪些不一致点

当前还剩的主要不一致点有两类。

### 9.1 非 DBLP 的遗留静态约束池仍未彻底收口

如果用户手动启用这些静态池，它们中的一部分 consequent 仍不是单边 `c`。  
这与当前“所有约束统一为 `(P, c)` 且 consequent 为单边”这一最终要求不完全一致。

### 9.2 主实验链之外的辅助脚本仍会各自构造 observed graph

当前主实验默认 runner 已经做到了“一次生成、全链复用”，但以下辅助脚本仍有各自的 masking 入口：

- case study 脚本
- tuning / sensitivity 脚本
- 若干比较脚本

这不影响主实验默认链，但说明“整个仓库所有入口都共用同一 observed graph 生成器”这件事还没完全做到。

---

## 10. 下一步最小修复建议

如果目标是把**整个仓库**都清理到只剩唯一主链，下一步最小修复建议是：

1. **清空或删除非 DBLP 的遗留静态约束池**
   - 至少先把 consequent 不是单边 `c` 的那 27 条遗留静态约束移出主仓可执行路径
   - 若以后要保留，只能重写成严格 `(P, c)` 形式后再恢复

2. **把辅助脚本统一接入 observed graph 共享入口**
   - 让 case study、tuning、sensitivity 等脚本也复用与主实验 runner 相同的 observed graph cache / preparation 函数

3. **在 README 或实验说明里明确 canonical entrypoints**
   - 明确指出：
     - 哪些 runner 才是唯一标准主链
     - 其余脚本仅为辅助分析，不作为正式实验口径

---

## 11. 最终判断

针对最核心的问题：

> “约束统一写成 `(P, c)`；只使用 antecedent / consequent；标准 backchase 明确是 `c -> P`；frequent pattern mining 已经保证 `c` 是真实存在、可匹配、非虚拟的；并且同一个 workload 的 observed graph 在整个实验链路中保持一致，不会被重复重新 masking。”

当前可以给出分层结论：

### 对主实验默认链路

**基本满足。**

- 术语已经统一
- 标准 backchase 已经恢复为 `c -> P`
- DBLP 默认 mined consequent 已有可匹配性筛选
- observed graph 在主实验 runner 中已经一次生成、全链复用
- baseline 与自有方法的 grounding 差异也已经明确

### 对整个仓库所有历史路径

**尚未完全满足。**

主要因为：

- 仍有非 DBLP 遗留静态约束池未彻底清空
- 仍有辅助脚本没有接入统一 observed graph 共享入口

因此，当前最准确的结论是：

- **可以把当前主实验默认链路当作后续实验的唯一主链**
- **但若要宣称“整个仓库已经彻底只剩这一版主链”，还需要再做一轮遗留静态约束与辅助脚本的收口**
