# witness定义与window入窗逻辑修复结果

## 1. 旧问题是什么

本轮修复前，项目里存在一个严重的统计与语义污染：

- `candidate_count > 0`
- `verified_count > 0`
- 但由于 `selected_witness_count = 0`
- 日志和部分输出会直接显示成 `witnesses=0`

这会把下面两件本来应该严格区分的事情混为一谈：

1. `candidate` 是否通过 verification，成为合法 `witness`
2. 该 `witness` 是否最终进入 window，或者是否最终 cover 到某条约束

按我们已经确认的方法定义：

- `witness = verified candidate`
- `selected witness` 是后续属性
- `covered constraint` 也是后续属性

因此，旧实现中把 `selected` 或 `covered` 误当作 `witness`，属于主链正确性错误。

## 2. 旧代码里原来把哪几层混淆了

本轮严审和修复确认，旧实现混淆了以下三层：

1. `verified witness`
2. `selected witness`
3. `covered constraint`

具体污染方式是：

- candidate 先生成
- candidate 也已经通过 verification
- 但如果当前 candidate 没有带来 grounded coverage，或者没被放进 window
- 日志层就会把它表现成 `witnesses=0`

这违反了我们当前的定义，因为：

- `coverage=0` 不等于 `witness=0`
- `selected=0` 也不等于 `witness=0`

## 3. 本轮修了哪些文件

### 3.1 主方法窗口逻辑

- [apxchase.py](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/apxchase.py)
- [heuchase.py](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/heuchase.py)
- [exhaustchase.py](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/exhaustchase.py)

修复点：

- 当 `window` 未满时，任何 verified witness 都允许先入窗
- 不再要求它必须先带来 grounded coverage 才能入窗
- 只有在 `window` 已满后，才让 replacement 逻辑继续要求改进 set objective

关键代码位置：

- [apxchase.py:1378](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/apxchase.py:1378)
- [heuchase.py:691](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/heuchase.py:691)
- [exhaustchase.py:702](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/exhaustchase.py:702)

### 3.2 节点 / 图实验输出

- [Run_Experiment_Node.py:199](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:199)
- [Run_Experiment.py:276](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:276)
- [Run_Experiment.py:630](/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:630)

修复点：

- 明确分开统计：
  - `candidate_count`
  - `verified_witness_count`
  - `selected_witness_count`
  - `covered_constraint_count`
- `num_witnesses` 兼容字段现在明确等于 `verified_witness_count`
- 终端日志里的 `witnesses=` 现在打印的是 `verified_witness_count`
- 若要看 window 内数量，会单独打印 `selected_witnesses=...`

### 3.3 bundle / collect 输出链

- [run_experiment_bundle.py:202](/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_experiment_bundle.py:202)
- [collect_results.py:149](/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/collect_results.py:149)

修复点：

- per-workload CSV 现在显式输出：
  - `candidate_count`
  - `verified_witness_count`
  - `selected_witness_count`
  - `covered_constraint_count`
- method summary 现在显式聚合：
  - `avg_candidate_count`
  - `avg_verified_witness_count`
  - `avg_selected_witness_count`
  - `avg_covered_constraint_count`
- 兼容字段 `witness_count` / `avg_witness_count` 现在都明确指向 verified witness，不再混入 selected 或 covered

## 4. witness 定义现在是否已经正确

现在已经对齐到我们要求的定义：

- `witness = verified candidate`

也就是说：

- 只要 candidate 通过 verification
- 它就应该计入 `verified_witness_count`

它后面是否：

- 被放进 window
- cover 到约束
- 进入最终 top-K set

都不会再反向污染 `witness` 的定义。

这条定义现在已经同时落实在：

- 终端日志
- 单 workload metrics JSON
- bundle per-workload CSV
- method summary CSV
- collect results 汇总链

## 5. window 未满时是否允许 verified witness 入窗

现在允许，而且这是本轮修复的关键。

当前行为是：

- 若 `window` 未满
- 那么 verified witness 会先被 admit 进入 `W_k`
- 即使它当前 `covered_constraint_count = 0`
- 也不会被拒绝

这保证了：

- 先有一个合法的 feasible witness set
- 再让后续 replacement / objective improvement 发挥作用

因此，当前不会再出现这种错误：

- verified witness 明明存在
- 但因为没有 coverage 就被 window 层直接拒绝

## 6. 修复前后典型 target 对比

以下对比基于：

- [witness_definition_fix_per_workload.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/witness_definition_fix_per_workload.csv)
- [witness_definition_fix_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/witness_definition_fix_summary.csv)

### 6.1 target 1101

#### 修复前结论

严审时真实情况已经确认：

- `candidate_count > 0`
- `verified_count > 0`
- `selected_witness_count = 0`
- `covered_constraint_count = 0`

但日志会错误表现成：

- `witnesses=0`

#### 修复后

`ApxC / 1101`

- `candidate_count = 48`
- `verified_witness_count = 48`
- `selected_witness_count = 6`
- `covered_constraint_count = 0`

`HeuC / 1101`

- `candidate_count = 20`
- `verified_witness_count = 20`
- `selected_witness_count = 6`
- `covered_constraint_count = 0`

`Exh / 1101`

- `candidate_count = 48`
- `verified_witness_count = 48`
- `selected_witness_count = 6`
- `covered_constraint_count = 0`

这证明：

- coverage 仍可为 `0`
- 但 witness 不再被错误记成 `0`

### 6.2 target 1262

`ApxC / 1262`

- `48 / 48 / 6 / 0`

`HeuC / 1262`

- `20 / 20 / 6 / 0`

`Exh / 1262`

- `48 / 48 / 6 / 0`

同样说明：

- `verified witness` 存在
- `selected witness` 也存在
- 只是当前 `covered_constraint_count = 0`
- 不应再被打印成 `witnesses=0`

### 6.3 正控制 target 1000

`ApxC / 1000`

- `candidate_count = 48`
- `verified_witness_count = 48`
- `selected_witness_count = 6`
- `covered_constraint_count = 4`

`HeuC / 1000`

- `candidate_count = 20`
- `verified_witness_count = 20`
- `selected_witness_count = 6`
- `covered_constraint_count = 4`

`Exh / 1000`

- `candidate_count = 48`
- `verified_witness_count = 26`
- `selected_witness_count = 6`
- `covered_constraint_count = 4`

这说明修复并没有破坏原来那些“既有 witness、也有 coverage”的正常 workload。

## 7. 现在是否还能出现“实际上有 verified witness，但日志显示 witness=0”

在当前已修复的主链路径上，不应该再出现。

原因是：

- 日志 `witnesses=` 已经改成打印 `verified_witness_count`
- CSV 的 `witness_count` / `num_witnesses` 也已经对齐到 verified 层
- `selected_witness_count` 和 `covered_constraint_count` 都已独立输出

因此，即使某个 workload 出现：

- `selected_witness_count = 0`
- 或 `covered_constraint_count = 0`

也不会再被错误显示成：

- `witnesses=0`

## 8. 修复后最小 bundle 验证

基于：

- [witness_bundle_smoke_per_workload.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/witness_bundle_smoke_per_workload.csv)
- [witness_bundle_smoke_method_summary.csv](/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/witness_bundle_smoke_method_summary.csv)

当前 bundle 输出已经能正确区分：

- `verified_witness_count`
- `selected_witness_count`
- `covered_constraint_count`

例如 `DBLP + gcn2 + {1101,1262}` 这组最小验证里：

`ApxC`

- `avg_verified_witness_count = 48.0`
- `avg_selected_witness_count = 6.0`
- `avg_covered_constraint_count = 0.0`

`HeuC`

- `avg_verified_witness_count = 20.0`
- `avg_selected_witness_count = 6.0`
- `avg_covered_constraint_count = 0.0`

这已经满足当前主链定义：

- witness 存在
- selection 单独统计
- coverage 单独统计

## 9. 当前结论

本轮修复已经把系统修到下面这个状态：

1. `witness` 严格等于 `verified candidate`
2. `selected witness` 与 `covered constraint` 已和 `witness` 明确分层
3. `window` 未满时，verified witness 可以进入 `window`
4. `coverage=0` 不会再被错误表现成 `witness=0`
5. 当前主链定义与日志/CSV 统计已经一致

一句话总结：

> 之前的问题不是“没有 witness”，而是“日志和 CSV 把 selected/covered 错当成了 witness”。这一点现在已经修正。
