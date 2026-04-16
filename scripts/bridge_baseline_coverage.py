#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from baselines import run_gnn_explainer_graph, run_gnn_explainer_node, run_pgexplainer_node, PGExplainerBaseline
from constraint_mining import resolve_constraints
from Edge_masking import mask_edges_by_constraints, mask_edges_for_node_classification
from experiment_common import build_run_config, load_yaml
from grounding_semantics import constraint_activation_summary, evaluate_grounding
from matcher import backchase_repair_cost, find_pattern_matches
from Run_Experiment import (
    _load_trained_model as _load_graph_model,
    _prepare_graph_for_model,
    _select_test_graph,
)
from Run_Experiment_Node import (
    _load_trained_model as _load_node_model,
)
from utils import dataset_func, set_seed


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "bridge_baseline_coverage.csv"
OUT_NOTES = RESULTS_DIR / "bridge_baseline_coverage_notes.txt"

DATASETS_IN_ORDER = ["MUTAG", "DBLP", "ATLAS", "Cora"]
EXPLAINERS = [("GNNExplainer", "gnnexplainer"), ("PGExplainer", "pgexplainer")]

BUNDLE_BY_DATASET = {
    "MUTAG": ROOT / "configs" / "bundles" / "manual_mutag.yaml",
    "DBLP": ROOT / "configs" / "bundles" / "manual_dblp.yaml",
    "Cora": ROOT / "configs" / "bundles" / "manual_cora.yaml",
}


def _manual_before_values() -> Dict[Tuple[str, str], Optional[float]]:
    data_json = ROOT / "scripts" / "manual_paper_tables.json"
    with data_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    figure7 = data["figure_7"]
    xs = list(figure7["x"])
    series = figure7["series"]
    out: Dict[Tuple[str, str], Optional[float]] = {}
    for explainer in ("GNNExplainer", "PGExplainer"):
        vals = series[explainer]
        for dataset, value in zip(xs, vals):
            out[(explainer, dataset)] = None if value is None else float(value)
    return out


def _load_bundle_context(dataset_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    bundle_path = BUNDLE_BY_DATASET[dataset_name]
    bundle_cfg = load_yaml(bundle_path)
    default_cfg = load_yaml(ROOT / "configs" / "local" / "default.yaml")
    dataset_key = str(bundle_cfg["dataset"]).lower()
    dataset_cfg = load_yaml(ROOT / "configs" / "local" / f"{dataset_key}.yaml")
    dataset_cfg["slug"] = dataset_key
    exp_cfg = {"name": str(bundle_cfg["run_name"]), "group": "bundle", "base": {k: v for k, v in bundle_cfg.items() if k not in {
        "dataset", "model_name", "methods", "factor_name", "factor_values", "output_root",
        "run_name", "reuse_observed_graph_cache", "default_config_path", "dataset_config_dir",
        "force", "base_overrides"
    }}}
    return bundle_cfg, default_cfg, dataset_cfg, exp_cfg


def _copy_witness_mappings(witness: Data, workload_graph: Data) -> None:
    if hasattr(workload_graph, "_nodes_in_full"):
        witness._nodes_in_full = getattr(workload_graph, "_nodes_in_full")
    if hasattr(workload_graph, "_nodes_in_observed"):
        witness._nodes_in_observed = getattr(workload_graph, "_nodes_in_observed")


def _prepare_observed_node_workload_uncached(
    data: Data,
    target_node: int,
    constraints: List[dict],
    config: Dict[str, Any],
) -> Data:
    observed_subgraph, _, _ = mask_edges_for_node_classification(
        data,
        target_node,
        constraints,
        num_hops=int(config.get("L", 2)),
        max_masks=int(config.get("max_masks", 1)),
        mask_ratio=config.get("mask_ratio", config.get("incompleteness")),
        seed=config.get("random_seed"),
        preserve_connectivity=bool(config.get("preserve_connectivity", True)),
    )
    return observed_subgraph


def _prepare_observed_graph_workload_uncached(
    pos: int,
    dataset_resource: Dict[str, Any],
    dataset: Any,
    constraints: List[dict],
    config: Dict[str, Any],
) -> Data:
    graph, _ = _select_test_graph(dataset_resource, dataset, pos)
    base_graph = _prepare_graph_for_model(graph)
    observed_graph, _ = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=int(config.get("max_masks", 1)),
        mask_ratio=config.get("mask_ratio", config.get("incompleteness")),
        seed=config.get("random_seed"),
        preserve_connectivity=bool(config.get("preserve_connectivity", True)),
    )
    observed_graph._clean = base_graph.clone()
    if hasattr(graph, "y") and graph.y is not None:
        observed_graph.y = graph.y.clone()
    observed_graph.E_base = observed_graph.edge_index.size(1)
    return observed_graph


def _node_witness_from_baseline(
    explainer_key: str,
    model: torch.nn.Module,
    observed_subgraph: Data,
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[Data]:
    target_id = int(observed_subgraph.target_node_subgraph_id)
    subgraph = observed_subgraph.clone().to(device)
    if explainer_key == "gnnexplainer":
        result = run_gnn_explainer_node(
            model=model,
            data=subgraph,
            target_node=target_id,
            epochs=config.get("gnn_epochs", 100),
            device=device,
        )
    else:
        result = run_pgexplainer_node(
            model=model,
            data=subgraph,
            target_node=target_id,
            epochs=config.get("pg_epochs", 30),
            device=device,
            full_data=subgraph,
        )
    edge_mask = result.get("edge_mask")
    if edge_mask is None or edge_mask.numel() == 0:
        return None
    k = min(int(config.get("baseline_edge_topk", config.get("K", config.get("k", 10)))), int(edge_mask.size(0)))
    _, topk_indices = torch.topk(edge_mask, k=k)
    expl_edge_index = subgraph.edge_index[:, topk_indices].cpu()
    witness = Data(
        x=subgraph.x.detach().cpu(),
        edge_index=expl_edge_index,
        y=subgraph.y.detach().cpu() if hasattr(subgraph, "y") and subgraph.y is not None else None,
        num_nodes=int(subgraph.num_nodes),
    )
    if hasattr(subgraph, "y_type") and subgraph.y_type is not None:
        witness.y_type = subgraph.y_type.detach().cpu()
    if hasattr(subgraph, "node_labels") and subgraph.node_labels is not None:
        witness.node_labels = subgraph.node_labels.detach().cpu()
    witness.target_node_subgraph_id = target_id
    _copy_witness_mappings(witness, observed_subgraph.cpu())
    return witness


def _graph_witness_from_baseline(
    explainer_key: str,
    model: torch.nn.Module,
    observed_graph: Data,
    config: Dict[str, Any],
    device: torch.device,
    pg_state: Dict[str, Any],
) -> Optional[Data]:
    masked_graph = observed_graph.clone().to(device)
    if explainer_key == "gnnexplainer":
        result = run_gnn_explainer_graph(
            model=model,
            graph=masked_graph,
            epochs=config.get("gnnexplainer_epochs", 100),
            device=device,
        )
        topk = int(config.get("gnnexplainer_topk", 10))
    else:
        if pg_state.get("explainer") is None:
            pg = PGExplainerBaseline(
                model,
                epochs=config.get("pgexplainer_epochs", 20),
                lr=config.get("pgexplainer_lr", 0.003),
                device=device,
            )
            pg_state["explainer"] = pg
        else:
            pg = pg_state["explainer"]
        if not pg_state.get("fitted", False):
            _ = pg.explain_graph(masked_graph, quick_fit=True)
            pg_state["fitted"] = True
        result = pg.explain_graph(masked_graph)
        topk = int(config.get("pgexplainer_topk", 10))
    edge_mask = result.get("edge_mask")
    if edge_mask is None or edge_mask.numel() == 0:
        return None
    edge_mask_flat = edge_mask.flatten()
    k = min(topk, int(edge_mask_flat.numel()))
    topk_indices = torch.topk(edge_mask_flat, k=k).indices
    witness = Data(
        x=masked_graph.x.detach().cpu(),
        edge_index=masked_graph.edge_index[:, topk_indices].detach().cpu(),
        batch=masked_graph.batch.detach().cpu() if hasattr(masked_graph, "batch") and masked_graph.batch is not None else None,
        y=masked_graph.y.detach().cpu() if hasattr(masked_graph, "y") and masked_graph.y is not None else None,
        num_nodes=int(masked_graph.num_nodes),
    )
    _copy_witness_mappings(witness, observed_graph.cpu())
    return witness


def _after_grounding_ratio(witness: Data, workload_graph: Data, constraints: List[dict], budget: int) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    workload_cpu = workload_graph.cpu()
    witness_cpu = witness.cpu()
    activation = constraint_activation_summary(workload_cpu, constraints, workload_cpu, find_pattern_matches)
    grounded = evaluate_grounding(
        witness_cpu,
        constraints,
        budget,
        observed_graph=workload_cpu,
        find_pattern_matches_fn=find_pattern_matches,
        backchase_repair_cost_fn=backchase_repair_cost,
    )
    active_names = sorted(activation["active_names"])
    ratio = (len(grounded) / len(active_names)) if active_names else 0.0
    return float(ratio), {
        "active_constraint_count": len(active_names),
        "active_constraints": active_names,
    }, {
        "covered_constraint_count": len(grounded),
        "covered_constraints": sorted(grounded),
    }


def _run_supported_dataset(dataset_name: str, explainer_name: str, explainer_key: str) -> float:
    bundle_cfg, default_cfg, dataset_cfg, exp_cfg = _load_bundle_context(dataset_name)
    method_key = explainer_key
    run_cfg = build_run_config(default_cfg, dataset_cfg, exp_cfg, method_key, str(bundle_cfg["model_name"]), {})
    if dataset_cfg["task"] == "graph":
        run_cfg["graph_positions"] = list(bundle_cfg.get("graph_positions", []))
    else:
        run_cfg["target_nodes"] = list(bundle_cfg.get("target_nodes", []))
    run_cfg["run_all"] = True

    set_seed(int(run_cfg.get("random_seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_resource = dataset_func(run_cfg)
    model = (_load_graph_model if dataset_cfg["task"] == "graph" else _load_node_model)(run_cfg, device)
    constraints = resolve_constraints(run_cfg, dataset_resource, model=model, device=device, save_dir=None)
    budget = int(run_cfg.get("local_budget_k", run_cfg.get("Budget", 2)))

    coverage_values: List[float] = []
    if dataset_cfg["task"] == "node":
        data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
        for target_node in run_cfg["target_nodes"]:
            observed_subgraph = _prepare_observed_node_workload_uncached(data, int(target_node), constraints, run_cfg)
            witness = _node_witness_from_baseline(explainer_key, model, observed_subgraph, run_cfg, device)
            if witness is None:
                coverage_values.append(0.0)
                continue
            ratio, _, _ = _after_grounding_ratio(witness, observed_subgraph, constraints, budget)
            coverage_values.append(ratio)
    else:
        dataset = dataset_resource["dataset"]
        pg_state: Dict[str, Any] = {"explainer": None, "fitted": False}
        for pos in run_cfg["graph_positions"]:
            observed_graph = _prepare_observed_graph_workload_uncached(int(pos), dataset_resource, dataset, constraints, run_cfg)
            witness = _graph_witness_from_baseline(explainer_key, model, observed_graph, run_cfg, device, pg_state)
            if witness is None:
                coverage_values.append(0.0)
                continue
            ratio, _, _ = _after_grounding_ratio(witness, observed_graph, constraints, budget)
            coverage_values.append(ratio)

    if not coverage_values:
        raise RuntimeError(f"No workloads processed for {dataset_name} / {explainer_name}")
    return float(sum(coverage_values) / len(coverage_values))


def main() -> None:
    before = _manual_before_values()
    rows: List[Dict[str, Any]] = []
    note_lines: List[str] = []
    note_lines.append("Bridge baseline coverage supplementary experiment")
    note_lines.append("")
    note_lines.append("Before-grounding source:")
    note_lines.append(f"- {ROOT / 'scripts' / 'manual_paper_tables.json'} :: section figure_7")
    note_lines.append(f"- Confirmed plotting entry point: {ROOT / 'scripts' / 'plot_manual_paper_figures.py'} reads DATA_JSON directly from that file.")
    note_lines.append("")
    note_lines.append("Checked files:")
    note_lines.append(f"- {ROOT / 'scripts' / 'manual_paper_tables.json'}")
    note_lines.append(f"- {ROOT / 'scripts' / 'plot_manual_paper_figures.py'}")
    note_lines.append(f"- {ROOT / 'src' / 'Run_Experiment.py'}")
    note_lines.append(f"- {ROOT / 'src' / 'Run_Experiment_Node.py'}")
    note_lines.append(f"- {ROOT / 'src' / 'utils.py'}")
    note_lines.append(f"- {ROOT / 'src' / 'grounding_semantics.py'}")
    note_lines.append("")
    note_lines.append("Before-grounding consistency check:")
    note_lines.append("- The current paper figures are driven by scripts/manual_paper_tables.json; the bridge table reuses those exact values verbatim.")
    note_lines.append("- No existing paper figure, CSV, or raw result file was modified.")
    note_lines.append("")
    note_lines.append("After-grounding computation:")
    note_lines.append("- For each supported dataset, reran the baseline explainer independently on the same bundle config/workload selection used by the current manual experiments.")
    note_lines.append("- Built witness G_s exactly as current baseline code does (same edge-mask top-k selection).")
    note_lines.append("- Then applied evaluate_grounding(...) with the same Σ and local budget B=local_budget_k used by our methods.")
    note_lines.append("- Normalized coverage after grounding was computed as |grounded constraints| / |active constraints on the observed workload graph|.")
    note_lines.append("")
    note_lines.append("Uncertainty / blockers:")
    note_lines.append("- ATLAS has before-grounding values in the manual paper table, but the repository does not contain an ATLAS dataset directory, config, model checkpoint, constraint pool, or cached run outputs.")
    note_lines.append("- Therefore ATLAS after-grounding could not be computed without fabricating assets; it is left blank.")

    for dataset in DATASETS_IN_ORDER:
        for explainer_name, explainer_key in EXPLAINERS:
            before_value = before.get((explainer_name, dataset))
            after_value: Optional[float]
            status = "ok"
            if dataset == "ATLAS":
                after_value = None
                status = "missing_dataset_assets"
            else:
                after_value = _run_supported_dataset(dataset, explainer_name, explainer_key)
            rows.append(
                {
                    "explainer": explainer_name,
                    "dataset": dataset,
                    "coverage_before": "" if before_value is None else f"{before_value:.6f}",
                    "coverage_after": "" if after_value is None else f"{after_value:.6f}",
                    "status": status,
                }
            )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["explainer", "dataset", "coverage_before", "coverage_after", "status"])
        writer.writeheader()
        writer.writerows(rows)

    OUT_NOTES.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_NOTES}")


if __name__ == "__main__":
    main()
