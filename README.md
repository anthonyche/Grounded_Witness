# GroundingGEXP

GroundingGEXP is a research codebase for studying graph explanations grounded by data constraints.  
The repository currently supports node-level and graph-level experiments on:

- `DBLP`
- `Cora`
- `MUTAG`

## What this repository contains

The codebase includes:

- implementations of our explanation methods:
  - `ApxC`
  - `HeuC`
  - `Exh`
- baseline explainers:
  - `GEX`
  - `PGX`
- constraint mining and constraint-aware evaluation
- unified experiment execution, aggregation, and plotting utilities

## Installation

Recommended environment:

- Python `3.10`
- PyTorch `2.3+`
- PyTorch Geometric `2.6+`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Main experiment entry point

The recommended way to run experiments is:

```bash
python scripts/run_experiment_bundle.py --config config.yaml
```

This entry point:

- reads a single YAML config
- prepares the observed graph for each workload
- runs multiple methods on the same workload set
- writes workload-level and method-level CSV summaries

## Configuration

The root `config.yaml` file is the default manual experiment template.

The most important fields are:

- `dataset`
- `model_name`
- `methods`
- `sigma_size`
- `local_budget_k`
- `L`
- `incompleteness`
- `target_ratio`
- `target_nodes` or `graph_positions`

Ready-to-edit templates are also provided in:

- `configs/bundles/manual_dblp.yaml`
- `configs/bundles/manual_cora.yaml`
- `configs/bundles/manual_mutag.yaml`

## Running an experiment

Example:

```bash
python scripts/run_experiment_bundle.py --config config.yaml --force
```

Use `--force` when you want to overwrite existing outputs for the same run name.

## Outputs

The main outputs are written to:

- `outputs/csv/<run_name>_per_workload.csv`
- `outputs/csv/<run_name>_method_summary.csv`
- `outputs/csv/<run_name>_metadata.json`

These files provide:

- workload-level metrics
- method-level aggregated summaries
- run configuration metadata

## Repository structure

```text
GroundingGEXP/
├── config.yaml
├── configs/
├── scripts/
│   ├── run_experiment_bundle.py
│   ├── experiment_common.py
│   ├── collect_results.py
│   └── plot_full_experiments.py
├── src/
├── models/
└── outputs/
```

## Notes

- Pretrained checkpoints are expected under `models/`.
- The current repository keeps a single recommended experiment entry path through `run_experiment_bundle.py`.
