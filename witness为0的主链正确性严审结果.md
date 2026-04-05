# witness=0 的主链正确性严审结果

## 1. 审计基准

本次严审严格按当前已经确认的方法定义进行，不按当前日志名义解释：

- `candidate`：生成出来的候选子图
- `verified candidate`：通过 `verify_witness` 的候选
- `witness`：按方法定义，应当等于 `verified candidate`
- `selected witness`：进入最终 window / `W_k` 的候选
- `covered witness`：其 grounding 能实际 cover 约束的 witness

按这个定义：

1. `witness` 不要求先 cover constraint
2. `witness` 不要求先进入最终 window
3. `coverage = 0` 不等于 `witness = 0`
4. backchase 是 witness 之后的过程，不应反过来定义 witness

因此，如果当前日志里出现：

- `witnesses = 0`
- `coverage = 0`
- `fid = 0`
- `conc = 0`

首先必须怀疑：

- 是不是代码把 `verified witness`、`selected witness`、`covered witness` 混成了一个统计口径

## 2. 结论先说

结论很明确：

1. 当前代码里，`witnesses=...` **不是**“所有 verified candidates 的数量”。
2. 当前 `witnesses=...` 统计的实际上是：**最终进入 window 的 selected witnesses 数量**。
3. 对这次审计抽查到的 `DBLP zero-witness` workload，主因不是：
   - 没有 candidate
   - 也不是 verification 全挂
4. 主因是：
   - `candidate_count > 0`
   - `verified_count > 0`
   - 但 `selected_count = 0`
   - 同时 `covered_count = 0`
5. 因而当前日志里的 `witnesses=0` 在语义上是**误报/误命名**：
   - 它表达的是“最终没有 selected witness”
   - 不是“真的没有 verified witness”

这违反了我们当前对 witness 的方法定义。

## 3. 代码里 `witness_count` 实际统计的是什么

### 3.1 Node 路径

在 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:188-275`：

- `result = chaser.explain_node(...)`
- `Sigma_star, witnesses = result`
- `metrics["num_witnesses"] = len(witnesses)`
- 日志也直接打印 `witnesses={len(witnesses)}`

关键点在于：

- 这里的 `witnesses` 完全依赖 `explain_node()` 返回值
- 代码没有单独统计“所有 verified candidates”

### 3.2 Graph 路径

同样的问题在 graph 路径也存在：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:329-364`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment.py:650-680`

两处都把：

- `num_witnesses = len(witnesses)`

写进 metrics。

所以：

- node 任务和 graph 任务都在复用同一套错误统计含义

## 4. `witnesses` 在 chaser 返回值里到底是什么

### 4.1 ApxC

在 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/apxchase.py:1407-1508`：

- 每个 candidate 都先做 `verify_witness_fn(...)`
- `n_verified += 1` 表示 verified candidate 数量
- 之后才调用 `_update_window(W_k, Gs, covered)`
- 最终返回的是：
  - `Sigma_star`
  - `S_k`

其中：

- `S_k = [entry[2] for entry in sorted(W_k, ...)]`

所以 ApxC 返回的 `witnesses` 实际上是：

- **最终 window 内被选中的图**

不是：

- 全部 verified candidates

### 4.2 HeuC

在 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/heuchase.py:830-894`：

- 逻辑和 ApxC 一样
- 先 verify
- 再 `_update_window`
- 最终返回 `annotated`

而 `annotated` 来源于：

- `S_k = [entry[2] for entry in sorted(W_k, ...)]`

所以 HeuC 里的 `witnesses` 也是：

- **最终 selected window**

不是：

- 全部 verified candidates

### 4.3 直接结论

因此，当前代码中“witness”这个词已经被错误地混用成了：

- selected witness

而不是方法定义要求的：

- verified witness

## 5. candidate / verify / select / cover 四层是否混淆

答案是：**混淆了，而且混得比较严重。**

当前实现至少混在一起了以下几层：

1. `candidate_count`
2. `verified_count`
3. `selected_count`
4. `coverage_increasing_count`
5. `covered_constraint_count`

### 5.1 `_update_window(...)` 会直接把 coverage 失败反向作用到 witness 统计

在 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/apxchase.py:1328-1405`：

- `Gamma_G = self.gamma_fn(H_view, self.Sigma, self.B)`
- 如果 `len(Gamma_G) == 0`，直接：
  - `return covered`

这意味着：

- candidate 即使已经通过 verify
- 只要它没有 grounded constraint
- 就不会进入 `W_k`

于是后面：

- `len(witnesses) = len(S_k) = len(W_k)`

就会变成 `0`。

这就把：

- “coverage failure”

错误地表现成了：

- “witness failure”

### 5.2 fallback 也没有纠正这个问题

在 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/apxchase.py:1465-1471` 和 `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/heuchase.py:854-860`：

- 当 `W_k == 0` 时，会尝试把完整 observed graph `H` 当 fallback
- 但前提仍是：
  - `verify_witness_fn(...)` 通过
- 然后仍然要调用：
  - `_update_window(W_k, H, covered)`

因此：

- observed graph 本身即使能通过 verify
- 只要 `Gamma(H) == 0`
- 仍然不会被计入最终 `witnesses`

所以 fallback 并没有把“verified witness”与“covered witness”拆开。

### 5.3 `_last_run_stats["num_candidates_admitted"]` 也不是 selected_count

在 ApxC 和 HeuC 里：

- `n_admitted` 的增加条件是：
  - `len(covered) > len(old_covered)`

因此它统计的是：

- **coverage 真正增加过的 candidate 数量**

不是：

- 最终 selected witness 数量

这一点可从正例 workload 看出来：

- workload `1000`
- `selected_witnesses = 6`
- 但 `num_candidates_admitted = 1`

所以当前内部统计本身也没有把：

- selected
- covered-improving

区分干净。

## 6. 对 zero-witness workload 的逐层诊断

本次严审使用当前 `DBLP` 日志里出现 `witnesses=0` 的 target，重点抽查：

- `1101`
- `1262`

并用：

- `1000`

作为正例对照。

辅助文件：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/candidate_verify_select_cover_counts.csv`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/zero_witness_workload_diagnosis.csv`

### 6.1 ApxC：target 1101

实例级 probe 结果：

- `candidate_count = 48`
- `verified_count = 48`
- `admitted_count = 0`
- `selected_witness_count = 0`
- `covered_constraint_count = 0`
- `observed_verify = True`

并且 candidate 边数分布不是空的：

- `candidate_size_min = 2`
- `candidate_size_max = 16`
- `candidate_size_median = 13.5`

结论：

- candidate 明确生成了
- verification 明确也通过了
- 但因为没有 grounded constraints，window 最终为空
- 于是日志打印成 `witnesses=0`

### 6.2 ApxC：target 1262

probe 结果：

- `candidate_count = 48`
- `verified_count = 48`
- `admitted_count = 0`
- `selected_witness_count = 0`
- `covered_constraint_count = 0`
- `observed_verify = True`

同样说明：

- 不是没有 witness
- 是没有 selected/covered witness

### 6.3 HeuC：target 1101

probe 结果：

- `candidate_count = 20`
- `verified_count = 20`
- `admitted_count = 0`
- `selected_witness_count = 0`
- `covered_constraint_count = 0`
- `observed_verify = True`

这说明 HeuC 上同样不是 verification 崩了，而是：

- verified candidate 没有进最终 window

### 6.4 HeuC：target 1262

probe 结果：

- `candidate_count = 20`
- `verified_count = 20`
- `admitted_count = 0`
- `selected_witness_count = 0`
- `covered_constraint_count = 0`
- `observed_verify = True`

### 6.5 正例对照：target 1000

`ApxC`：

- `candidate_count = 48`
- `verified_count = 48`
- `admitted_count = 1`
- `selected_witness_count = 6`
- `covered_constraint_count = 4`
- `observed_verify = True`

`HeuC`：

- `candidate_count = 20`
- `verified_count = 20`
- `admitted_count = 1`
- `selected_witness_count = 6`
- `covered_constraint_count = 4`
- `observed_verify = True`

这个对照很关键，它说明：

- verification 基本上不是决定 zero-witness 的关键层
- 真正决定日志里 `witnesses` 是否为 0 的，是：
  - 是否进入 `W_k`
  - 是否带来 grounded coverage

## 7. 更大范围的全量 zero-witness 扫描

对 `manual_dblp_per_workload.csv` 里全部 `ApxC / HeuC zero-witness` workload 做了抽样外全量 probe。

辅助文件：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/zero_witness_workload_diagnosis.csv`

先看外层现象：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/manual_dblp_per_workload.csv`

其中：

- `ApxC` 有 `14` 个 zero-witness workload
- `HeuC` 也有 `14` 个 zero-witness workload
- 这 `28` 条记录全部满足：
  - `hit_consequent_constraint_count > 0`
  - `active_constraint_count = 0`
  - `covered_constraint_count = 0`

也就是说，从外层 CSV 看，这批 workload 本来就不是“什么都没发生”：

- `c` 已经命中
- 但没有进入 active，更没有 cover

然后再看内部 probe。

汇总结果：

### ApxC

- zero-witness workload 数：`14`
- `candidate_count`: 全部 `48`
- `verified_count`: `min=47, max=48, mean=47.93`
- `admitted_count`: 全部 `0`
- `selected_witness_count`: 全部 `0`
- `covered_constraint_count`: 全部 `0`

### HeuC

- zero-witness workload 数：`14`
- `candidate_count`: 全部 `20`
- `verified_count`: 全部 `20`
- `admitted_count`: 全部 `0`
- `selected_witness_count`: 全部 `0`
- `covered_constraint_count`: 全部 `0`

这已经足够说明：

- 在我们自己的方法上，当前 `witnesses=0` 的主流真实情况不是“没有 verified candidate”
- 而是“verified candidate 很多，但 selected/covered 为 0”

## 8. observed graph 本身能否通过 verify

对上面几个典型 zero-witness workload：

- `1101`
- `1262`

以及正例 `1000`，

probe 结果都是：

- `observed_verify = True`

也就是说：

- observed graph 本身作为 candidate reference 是能够通过当前 verify 的

这非常关键，因为它直接回答了一个尖锐问题：

> 如果 observed graph 本身都能过 verify，但日志仍然打印 `witnesses=0`，那么 `witnesses=0` 就不能再被解释成“没有 witness”。

答案是：

- 现在确实不能这样解释

## 9. baseline 的 `witness_count=0` 是另一类统计错误

baseline 路径在：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/src/Run_Experiment_Node.py:294-408`

这里：

- baseline 实际上会构建一个 explanation subgraph
- 并返回 `(elapsed, 1, fid_minus, conciseness, coverage)`

但写入 metrics 时：

- 没有保存 `num_witnesses`

而 bundle 汇总在：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_experiment_bundle.py:202`

使用：

- `metrics.get("num_witnesses", 0)`

于是 `GEX / PGX` 在 CSV 里会出现：

- `witness_count = 0`

即使同时还有：

- 非零 `fidelity_minus`
- 非零 `conciseness`

因此 baseline 的 `witness_count=0` 是纯粹的：

- metrics plumbing/export bug

而不是算法现象。

## 10. 对核心问题的最终回答

### Q1. 对于 `witnesses=0` 的 workload，真实情况到底是哪一种？

答案是：

- **E. 多种情况混合存在**

但对当前审到的自有方法主案例，主流模式很明确：

- `candidate_count > 0`
- `verified_count > 0`
- `selected_count = 0`
- `covered_count = 0`

也就是：

- **C + D 的链式情况**

更准确地说：

- 先有 verified witness
- 但因为没有 grounded coverage，它们不被计入最终 selected window
- 所以日志被打印成 `witnesses=0`

### Q2. 当前代码里，“witness”这个词是否被错误地混用了？

答案是：

- **是，明显被混用了。**

至少混淆了：

- verified witness
- selected witness
- covered / grounding-success witness

当前用户看到的：

- `witnesses=...`

实际统计的是：

- `selected_count`

而不是：

- `verified_count`

### Q3. 如果 observed graph 本身作为候选参考对象都能通过当前 verify，这是否意味着 verification 定义或 plumbing 有问题？

答案是：

- **是的，至少证明当前日志层的 witness 解释有问题。**

因为审计样本已经显示：

- observed graph `verify = True`
- 但最后 `witnesses = 0`

所以：

- verification 通过 ≠ 日志里的 witness 存在

这说明 witness 统计口径和方法定义不一致。

### Q4. 当前 `witnesses=0` 现象最主要是什么问题？

按重要性排序：

1. **witness_count 统计定义问题**
2. **window / coverage gating 污染了 witness 定义**
3. **grounding / coverage 失败被错误表现成 witness 失败**
4. verification 不是当前审计样本的主因
5. candidate generation 不是当前审计样本的主因

## 11. 最小修复建议（这一步先不改代码）

本次只严审，不实际改代码。按当前发现，最小修复应该是：

1. 明确把当前 `num_witnesses` 改名为：
   - `selected_witness_count`
2. 单独新增：
   - `verified_candidate_count`
3. 单独新增：
   - `coverage_increasing_candidate_count`
   - 不要继续用 `admitted` 这个模糊词
4. 日志必须至少同时打印：
   - `candidates`
   - `verified`
   - `selected`
   - `covered_constraints`
5. baseline 路径必须显式输出：
   - `num_witnesses = 1`
   或改成单独字段：
   - `explanation_count = 1`
6. 主报告里的“witness”必须恢复为：
   - verified candidate

## 12. 最终结论

最终结论一句话：

> 当前出现 `witnesses=0`，在我们自己的方法上，主要不是因为真的没有 candidate / witness，而是因为代码把 `verified candidate`、`selected window` 和 `coverage success` 这几层概念混淆了；当前日志里的 `witnesses=0` 实际上更接近“selected witness = 0”，这违反了我们当前对 witness 的方法定义。
