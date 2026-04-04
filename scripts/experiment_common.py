#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]

METHOD_SPECS = {
    "apxchase": {"exp_prefix": "apxchase", "label": "ApxC"},
    "heuchase": {"exp_prefix": "heuchase", "label": "HeuC"},
    "gnnexplainer": {"exp_prefix": "gnnexplainer", "label": "GEX"},
    "pgexplainer": {"exp_prefix": "pgexplainer", "label": "PGX"},
    "exhaustchase": {"exp_prefix": "exhaustchase", "label": "Exh"},
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _node_lhop_sizes(data: Any, test_nodes: List[int], l_hops: int) -> List[tuple[int, int]]:
    from torch_geometric.utils import k_hop_subgraph

    sizes: List[tuple[int, int]] = []
    for node in test_nodes:
        subset, _, _, _ = k_hop_subgraph(int(node), l_hops, data.edge_index, relabel_nodes=False)
        sizes.append((int(node), int(subset.numel())))
    return sizes


def _balanced_node_sample(
    data: Any,
    test_nodes: List[int],
    sample_size: int,
    l_hops: int,
    rng: random.Random,
    tail_percentile: float,
) -> List[int]:
    if sample_size >= len(test_nodes):
        return sorted(test_nodes)

    sized_nodes = sorted(_node_lhop_sizes(data, test_nodes, l_hops), key=lambda item: (item[1], item[0]))
    cutoff_rank = max(sample_size, math.ceil(len(sized_nodes) * tail_percentile))
    eligible = sized_nodes[:cutoff_rank]
    if len(eligible) < sample_size:
        eligible = sized_nodes

    chosen: List[int] = []
    for idx in range(sample_size):
        start = int(idx * len(eligible) / sample_size)
        end = int((idx + 1) * len(eligible) / sample_size)
        bucket = eligible[start:max(start + 1, end)]
        chosen.append(int(rng.choice(bucket)[0]))
    return sorted(chosen)


def select_targets(dataset_cfg: Dict[str, Any], dataset_resource: Any, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(run_cfg.get("random_seed", 0)))
    ratio = float(run_cfg.get("target_ratio", dataset_cfg.get("default_target_ratio", 1.0)))
    max_targets = run_cfg.get("max_targets")
    fixed_targets = run_cfg.get("target_nodes")
    fixed_graph_positions = run_cfg.get("graph_positions")

    if dataset_cfg["task"] == "node":
        if fixed_targets:
            return {"target_nodes": [int(v) for v in fixed_targets], "run_all": True}
        data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
        test_nodes = [int(v) for v in data.test_mask.nonzero(as_tuple=True)[0].tolist()]
        total_population = int(dataset_cfg.get("config", {}).get("data_size", getattr(data, "num_nodes", len(test_nodes))))
        sample_size = max(int(dataset_cfg.get("min_targets", 1)), int(total_population * ratio))
        if ratio > 0.0:
            sample_size = max(1, sample_size)
        if max_targets is not None:
            sample_size = min(sample_size, int(max_targets))
        sample_size = min(sample_size, len(test_nodes))
        sampling_strategy = str(run_cfg.get("target_sampling_strategy", "balanced_lhop"))
        if sampling_strategy == "balanced_lhop" and sample_size < len(test_nodes):
            l_hops = int(run_cfg.get("L", dataset_cfg.get("config", {}).get("L", 2)))
            tail_percentile = float(run_cfg.get("target_size_percentile_cap", 0.85))
            chosen = _balanced_node_sample(data, test_nodes, sample_size, l_hops, rng, tail_percentile)
        else:
            chosen = sorted(rng.sample(test_nodes, sample_size)) if sample_size < len(test_nodes) else sorted(test_nodes)
        return {"target_nodes": chosen, "run_all": True}

    if fixed_graph_positions:
        return {"graph_positions": [int(v) for v in fixed_graph_positions], "run_all": True}
    test_subset = dataset_resource["test_loader"].dataset
    positions = list(range(len(test_subset.indices)))
    total_population = int(dataset_cfg.get("config", {}).get("data_size", len(positions)))
    sample_size = max(int(dataset_cfg.get("min_targets", 1)), int(total_population * ratio))
    if ratio > 0.0:
        sample_size = max(1, sample_size)
    if max_targets is not None:
        sample_size = min(sample_size, int(max_targets))
    sample_size = min(sample_size, len(positions))
    chosen = sorted(rng.sample(positions, sample_size)) if sample_size < len(positions) else positions
    return {"graph_positions": chosen, "run_all": True}


def resolved_entry_script(dataset_cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> Path:
    script = exp_cfg.get("runner_script")
    if script:
        return ROOT / script
    if dataset_cfg["task"] == "graph":
        return ROOT / "src" / "Run_Experiment.py"
    return ROOT / "src" / "Run_Experiment_Node.py"


def build_run_config(
    default_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    method_key: str,
    model_key: str,
    combo: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged = deep_merge(merged, default_cfg.get("defaults", {}))
    merged = deep_merge(merged, dataset_cfg.get("config", {}))
    merged = deep_merge(merged, exp_cfg.get("base", {}))
    merged = deep_merge(merged, combo)

    if exp_cfg.get("tie_alpha_beta_to_gamma", False) and "gamma" in merged:
        alpha_beta = (1.0 - float(merged["gamma"])) / 2.0
        merged["alpha"] = alpha_beta
        merged["beta"] = alpha_beta

    resolved_model_key = str(merged.pop("model_key_override", model_key))
    if resolved_model_key not in dataset_cfg["models"]:
        raise ValueError(f"Unknown model key override '{resolved_model_key}' for dataset {dataset_cfg['data_name']}")

    merged["data_name"] = dataset_cfg["data_name"]
    merged["model_name"] = dataset_cfg["models"][resolved_model_key]
    merged["exp_name"] = f"{METHOD_SPECS[method_key]['exp_prefix']}_{exp_cfg['name']}_{dataset_cfg['slug']}"
    merged["K"] = int(merged.get("K", merged.get("k", 6)))
    merged["k"] = int(merged["K"])
    merged["local_budget_k"] = int(merged.get("local_budget_k", merged.get("Budget", 2)))
    merged["Budget"] = int(merged["local_budget_k"])
    merged["incompleteness"] = float(merged.get("incompleteness", merged.get("mask_ratio", 0.05)))
    merged["mask_ratio"] = float(merged["incompleteness"])
    merged["sigma_size"] = int(merged.get("sigma_size", merged.get("constraint_limit", 20)))
    merged["constraint_limit"] = int(merged["sigma_size"])
    merged["constraint_max_patterns"] = max(int(merged.get("constraint_max_patterns", merged["sigma_size"])), int(merged["sigma_size"]))
    merged["baseline_edge_topk"] = int(merged.get("baseline_edge_topk", merged["K"]))
    merged["method_key"] = method_key
    merged["method_label"] = METHOD_SPECS[method_key]["label"]
    merged["model_key"] = resolved_model_key
    merged["dataset_slug"] = dataset_cfg["slug"]
    return merged


def write_run_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_run_status(path: Path, status: Dict[str, Any]) -> None:
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
