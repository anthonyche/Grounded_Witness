# run_experiment_bundle 设计说明

## 1. 脚本入口

统一实验入口脚本是：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/scripts/run_experiment_bundle.py`

调用方式：

```bash
python scripts/run_experiment_bundle.py --config <bundle_config.yaml>
```

现在可以直接用这些现成模板：

- `/Users/anthonyche/Desktop/Research/GroundingGEXP/config.yaml`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/configs/bundles/manual_dblp.yaml`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/configs/bundles/manual_cora.yaml`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/configs/bundles/manual_mutag.yaml`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/configs/bundles/dblp_sigma_sweep.yaml`
- `/Users/anthonyche/Desktop/Research/GroundingGEXP/configs/bundles/dblp_incompleteness_sweep.yaml`

如需覆盖已有 raw 结果，可加：

```bash
python scripts/run_experiment_bundle.py --config <bundle_config.yaml> --force
```

这个入口的目标是：

- 一次只读一个 config
- 先前置生成并缓存 observed graph
- 对同一批 workload，让所有方法共享同一份 observed graph
- 自动输出 per-workload 明细、method-level summary 和 run metadata

## 2. config 字段

最小必填字段：

- `dataset`
- `model_name`
- `methods`
- `run_name`

推荐完整字段：

```yaml
dataset: dblp
model_name: gcn2
methods:
  - ApxC
  - HeuC
  - Exh
  - GEX
  - PGX

sigma_size: 20
local_budget_k: 2
L: 2
gamma: 1.0
alpha: 0.5
beta: 0.5
incompleteness: 0.05
target_ratio: 0.01
random_seed: 42

factor_name: sigma_size
factor_values: [10, 20, 30, 40, 50]

output_root: outputs/raw/bundles
run_name: dblp_sigma_bundle
reuse_observed_graph_cache: true
```

### 关于 workload 固定的强建议

如果你只是想在**同一次 bundle run**里保证所有方法公平共享同一套 workload，那么：

- 留 `target_ratio` / `max_targets`
- 不写 `target_nodes` / `graph_positions`

也可以，因为 bundle 会先采样一次，再让所有方法复用。

但如果你想在**多次手动改 config 之后仍保持完全同一套 workload**，就不要依赖 `target_ratio`。请显式写：

- node 任务：`target_nodes`
- graph 任务：`graph_positions`

这是为了避免你改了别的字段后又重新采样，导致前后实验不可比。

字段说明：

- `dataset`
  - 当前支持：`dblp` / `cora` / `mutag`
- `model_name`
  - 这里填的是当前数据集配置里的模型 key，例如 `gcn2`、`gcn3`、`gat2`
- `methods`
  - 支持大小写混用，会自动规范到：
    - `ApxC`
    - `HeuC`
    - `Exh`
    - `GEX`
    - `PGX`
- `factor_name`
  - 为空或缺省时，只跑一个默认点
- `factor_values`
  - 当 `factor_name` 非空时必须给列表
- `output_root`
  - raw 结果根目录
- `run_name`
  - 本次 bundle 的稳定名字，也会进入 CSV 文件名
- `reuse_observed_graph_cache`
  - `true` 时复用 `artifacts/observed_graph_cache/<dataset>/`
  - `false` 时每个 factor value 前先清掉该数据集的 observed graph cache 再重建

除上面这些脚本控制字段以外，其他顶层字段会直接透传到运行配置里。也就是说：

- `sigma_size`
- `local_budget_k`
- `L`
- `gamma`
- `alpha`
- `beta`
- `incompleteness`
- `target_ratio`
- `random_seed`
- `constraint_source`
- `constraint_type_source`
- `constraint_rule_mode`
- `constraint_filter_target_matchability`
- `constraint_pool_mode`
- `constraint_pool_file`
- `max_targets`

都可以直接在 bundle config 顶层写。

### 规则池切换

对于当前主线 `DBLP`，bundle config 现在还支持：

- `constraint_pool_mode: original`
- `constraint_pool_mode: balanced`
- `constraint_pool_mode: coverage_only`

默认仍然是 `original`，不会自动切到 tuned pool。

如果你想用 balanced 规则池，只需要在 config 里加一行：

```yaml
constraint_pool_mode: balanced
```

如果你想手动指定固定规则名单，也可以显式给文件：

```yaml
constraint_pool_file: configs/constraint_pools/dblp_balanced_p7_t0p4.yaml
```

这一步只改变 mined 后的规则选择，不改变：

- observed graph 共享机制
- candidate generation
- greedy / UpdateWK
- backchase 主语义

## 3. observed graph 如何缓存与共享

这是这个入口最重要的设计点。

### 当前保证

对同一个 factor value 下的同一批 workload：

- observed graph 只前置生成一次
- 然后 `ApxC / HeuC / Exh / GEX / PGX` 共享同一份 observed graph

当前实现方式是：

- bundle 先用 runner 现有的 observed graph cache 逻辑预热
- 后续各方法 runner 通过同一个 cache key 复用同一份 observed graph

所以它不是“靠相同 seed 大概率一样”，而是“同一路径、同一缓存键、同一份文件复用”。

### 缓存键

observed graph cache 使用 runner 现有键逻辑，显式包含：

- dataset
- workload id
- `L`
- `mask_ratio / incompleteness`
- constraint signature
- random seed

这样同一实验点下：

- 方法之间共享 cache
- 不同 factor value 不会误复用不兼容的 observed graph

### 重要语义

observed graph 的生成属于前置步骤：

- 不计入方法 runtime
- 只在 factor = `incompleteness` 时允许改变 observed graph
- 即使是 incompleteness study，也必须先前置生成同一份 observed graph，再让所有方法共享

## 4. runtime 定义

bundle 输出两层 runtime：

### 方法级 runtime

每个 workload × method 的：

- `runtime_method_only`

只计方法执行时间，不含 observed graph 预处理。

### 汇总 runtime

method-level summary 里输出：

- `runtime_total`
- `runtime_per_workload`

定义是：

- `runtime_total`
  - 该方法这一整次 run 的总运行时间
- `runtime_per_workload`
  - 该方法在已完成 workload 上的平均方法时间

不计入的方法时间：

- observed graph masking / cache 生成
- factor 前置预热

## 5. 输出文件

脚本每次至少会稳定输出三份文件：

- `outputs/csv/<run_name>_per_workload.csv`
- `outputs/csv/<run_name>_method_summary.csv`
- `outputs/csv/<run_name>_metadata.json`

### per-workload CSV

每行对应：

- 一个 workload
- 一个方法
- 一个 factor value

核心字段：

- `dataset`
- `model_name`
- `resolved_model_name`
- `method`
- `factor_name`
- `factor_value`
- `workload_id`
- `num_nodes_observed`
- `num_edges_observed`
- `witness_count`
- `hit_consequent_constraint_count`
- `active_constraint_count`
- `covered_constraint_count`
- `coverage_global`
- `coverage_normalized`
- `conciseness`
- `fidelity_minus`
- `runtime_method_only`
- `timeout_flag`
- `status`

### method-level summary CSV

按：

- dataset
- model
- method
- factor value

聚合，输出：

- `num_workloads`
- `num_completed`
- `timeout_count`
- `avg_witness_count`
- `avg_hit_consequent_constraint_count`
- `avg_active_constraint_count`
- `avg_covered_constraint_count`
- `avg_coverage_global`
- `avg_coverage_normalized`
- `avg_conciseness`
- `avg_fidelity_minus`
- `runtime_total`
- `runtime_per_workload`

### metadata JSON

记录：

- bundle config snapshot
- dataset / model / methods
- factor 设置
- resolved constraint names
- workload ids
- observed graph cache policy
- 每个 factor value 的预热 workload 规模信息

## 6. 这个入口如何取代旧的零散脚本

旧的链路是：

- 手工改很多 experiment YAML
- 跑零散脚本
- 再用单独汇总脚本去读 outputs/raw

新的 bundle 入口更适合人工手动实验：

1. 改一个 config
2. 跑一个脚本
3. 直接得到：
   - raw
   - per-workload CSV
   - method summary CSV
   - metadata JSON

它不替代算法 runner 本身：

- `Run_Experiment_Node.py`
- `Run_Experiment.py`

但会替代手工组织实验矩阵的零散入口。

## 7. 当前是否可以作为唯一主实验入口

对于当前主链数据集和方法，结论是：

- 可以作为**唯一手动主实验入口**

当前已验证：

- `DBLP` 上 bundle smoke 已跑通
- 预热的 workload id 与实际运行的 workload id 一致
- observed graph 确实是前置生成、方法复用
- 输出 CSV 与 metadata 结构稳定

当前适用范围：

- 数据集：`DBLP / Cora / MUTAG`
- 方法：`ApxC / HeuC / Exh / GEX / PGX`

当前推荐工作流：

1. 手动实验：直接改根目录 `/Users/anthonyche/Desktop/Research/GroundingGEXP/config.yaml`
2. 固定模板实验：从 `configs/bundles/*.yaml` 复制一份再改
3. 自动批量实验：继续保留 `configs/local/full/*.yaml` 和旧 pipeline，但后续建议逐步迁到 bundle 形态

如果后续要完全替代所有旧实验脚本，建议再做两件事：

1. 把现有 full/efficiency/effectiveness YAML 逐步迁到 bundle config 形态
2. 后续 plot 脚本优先读取 bundle 产出的 summary CSV，而不是再去扫零散 raw 目录
