# GroundingGEXP

GroundingGEXP is the current mainline repository for our constraint-grounded graph explanation experiments. The repository is organized around a single semantics-consistent experimental path:

- each constraint is written as `φ = (P, c)`
- `P` is the `antecedent`
- `c` is the `consequent`
- standard semantics:
  - `chase: P -> c`
  - `backchase: c -> P`

The current mainline datasets are:

- `DBLP`
- `Cora`
- `MUTAG`

## Mainline Semantics

### Our methods

The following methods first generate witness subgraphs `G_s`, then perform `G_s -> G_g` grounding / backchase:

- `ApxC`
- `HeuC`
- `Exh`

Under the current semantics:

- `G_g` is only an auxiliary grounded provenance graph
- `G_g` is never written back into the observed graph `G`
- completion may add edges, but may not add nodes

### Baselines

The following baselines do **not** perform `G_s -> G_g` backchase expansion:

- `GEX`
- `PGX`

Baselines only output a single explanation. A constraint is counted as covered only if that explanation itself strictly satisfies the constraint.

### Coverage

Coverage is computed per workload first, then averaged across workloads:

- `coverage_ratio_global = |Covered(Q)| / |Σ|`
- `coverage_ratio_normalized = |Covered(Q)| / |Active(Q)|`

where:

- `Hit_c(Q)`: constraints whose consequent `c` is matchable
- `Active(Q)`: constraints whose consequent `c` is matchable and whose antecedent `P` is node-complete
- `Covered(Q)`: constraints that can complete standard backchase within the local budget

## Recommended Experiment Entry Point

The only recommended experiment entry point is:

```bash
python scripts/run_experiment_bundle.py --config config.yaml
```

This entry point:

- reads a single YAML config
- precomputes the observed graph for each workload
- shares the same observed graph across all methods for that workload
- writes stable workload-level CSVs, method-level summaries, and metadata

Older fragmented runners are no longer recommended.

## Observed Graph Sharing and Runtime Definition

For the same workload:

- the observed graph is generated exactly once
- `ApxC / HeuC / Exh / GEX / PGX` all reuse it
- observed graph generation is treated as preprocessing
- method runtime does **not** include that preprocessing time

The observed graph is only allowed to change during an `incompleteness` factor study. Even there, the changed graph must be generated as a shared preprocessing step and must not be counted toward method runtime.

## Manual Experiment Configuration

The root `config.yaml` file is a neutral manual template. You can directly edit:

- `dataset`
- `model_name`
- `methods`
- `sigma_size`
- `local_budget_k`
- `L`
- `gamma / alpha / beta`
- `incompleteness`
- `target_ratio`
- `target_nodes` or `graph_positions`

Ready-to-edit bundle templates are also provided:

- `configs/bundles/manual_dblp.yaml`
- `configs/bundles/manual_cora.yaml`
- `configs/bundles/manual_mutag.yaml`

### Example: DBLP overall runtime

```bash
python scripts/run_experiment_bundle.py --config config.yaml --force
```

### Example: switching constraint pool mode

Add one of the following to `config.yaml`:

```yaml
constraint_pool_mode: balanced
```

Supported values:

- `original`
- `balanced`
- `coverage_only`

If omitted, the dataset default constraint pool is used.

## Output Files

The bundle runner writes at least:

- `outputs/csv/<run_name>_per_workload.csv`
- `outputs/csv/<run_name>_method_summary.csv`
- `outputs/csv/<run_name>_metadata.json`

The workload-level CSV explicitly separates:

- `candidate_count`
- `verified_witness_count`
- `selected_witness_count`
- `covered_constraint_count`

The current witness definition is:

- `witness = verified candidate`

Selected witnesses and covered constraints are reported separately.

## Environment

- Python `3.10`
- PyTorch `2.3+`
- PyTorch Geometric `2.6+`

Optional PyG extensions such as `torch-scatter`, `torch-sparse`, `torch-cluster`, and `torch-spline-conv` should match the installed PyTorch version. If they do not, the code may still run, but runtime measurements can be distorted.

## Repository Layout

```text
GroundingGEXP/
├── config.yaml
├── configs/
│   ├── bundles/
│   └── local/
├── scripts/
│   ├── run_experiment_bundle.py
│   ├── experiment_common.py
│   ├── collect_results.py
│   └── plot_full_experiments.py
├── src/
│   ├── Run_Experiment.py
│   ├── Run_Experiment_Node.py
│   ├── grounding_semantics.py
│   ├── constraint_mining.py
│   ├── constraints.py
│   ├── apxchase.py
│   ├── heuchase.py
│   ├── exhaustchase.py
│   ├── baselines.py
│   └── utils.py
├── models/
└── outputs/
```

## Current Status

The repository has been narrowed to the current mainline:

- constraint definition as `(P, c)`
- standard backchase `c -> P`
- shared observed graph
- `run_experiment_bundle.py` as the single recommended experiment entry point

Legacy large-scale experiments and older helper scripts are not part of the current default mainline.
