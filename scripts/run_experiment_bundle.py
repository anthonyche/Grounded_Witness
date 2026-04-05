#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_common import (  # type: ignore
    METHOD_SPECS,
    build_run_config,
    deep_merge,
    load_yaml,
    resolved_entry_script,
    select_targets,
    slugify,
    write_run_manifest,
    write_run_status,
)
METHOD_ALIASES = {
    "apxc": "apxchase",
    "apxchase": "apxchase",
    "heuc": "heuchase",
    "heuchase": "heuchase",
    "exh": "exhaustchase",
    "exhaustchase": "exhaustchase",
    "gex": "gnnexplainer",
    "gnnexplainer": "gnnexplainer",
    "pgx": "pgexplainer",
    "pgexplainer": "pgexplainer",
}

SCRIPT_RESERVED_KEYS = {
    "dataset",
    "model_name",
    "methods",
    "factor_name",
    "factor_values",
    "output_root",
    "run_name",
    "reuse_observed_graph_cache",
    "default_config_path",
    "dataset_config_dir",
    "force",
    "base_overrides",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment bundle from a single config.")
    parser.add_argument("--config", required=True, help="Path to bundle YAML.")
    parser.add_argument("--force", action="store_true", help="Re-run methods even if raw outputs already exist.")
    return parser.parse_args()


def _normalize_method_key(name: str) -> str:
    key = METHOD_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"Unsupported method name: {name}")
    return key


def _load_model_checkpoint(run_cfg: Dict[str, Any], device: torch.device) -> Optional[torch.nn.Module]:
    from model import get_model

    type_source = str(run_cfg.get("constraint_type_source", "")).lower()
    if type_source not in {"predicted", "predicted_label", "model_prediction"}:
        return None
    model = get_model(run_cfg).to(device)
    model_path = ROOT / "models" / f"{run_cfg['data_name']}_{run_cfg['model_name']}_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def _coerce_factor_values(bundle_cfg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    factor_name = str(bundle_cfg.get("factor_name") or "").strip()
    if not factor_name:
        return [("", None)]
    values = bundle_cfg.get("factor_values")
    if not isinstance(values, list) or not values:
        raise ValueError("factor_values must be a non-empty list when factor_name is provided.")
    return [(factor_name, value) for value in values]


def _bundle_base_overrides(bundle_cfg: Dict[str, Any]) -> Dict[str, Any]:
    overrides = dict(bundle_cfg.get("base_overrides", {}) or {})
    for key, value in bundle_cfg.items():
        if key in SCRIPT_RESERVED_KEYS:
            continue
        overrides[key] = value
    return overrides


def _clear_observed_cache(dataset_name: str) -> None:
    cache_dir = ROOT / "artifacts" / "observed_graph_cache" / dataset_name.lower()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _factor_token(factor_name: str, factor_value: Any) -> str:
    if not factor_name:
        return "default"
    return f"{slugify(factor_name)}__{slugify(factor_value)}"


def _bundle_run_dir(
    output_root: Path,
    run_name: str,
    dataset_slug: str,
    model_key: str,
    factor_name: str,
    factor_value: Any,
    method_key: str,
) -> Path:
    return output_root / run_name / dataset_slug / model_key / _factor_token(factor_name, factor_value) / method_key


def _metric_files_for_task(run_dir: Path, task: str) -> List[Path]:
    if task == "graph":
        return sorted(run_dir.glob("metrics_graph_*.json"))
    return sorted(run_dir.glob("metrics_node_*.json"))


def _load_observed_graph(run_dir: Path, task: str, workload_id: int) -> Optional[Any]:
    if task == "graph":
        path = run_dir / f"observed_graph_{int(workload_id)}.pt"
    else:
        path = run_dir / f"observed_subgraph_node_{int(workload_id)}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _per_workload_row(
    metric_path: Path,
    metrics: Dict[str, Any],
    manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
    factor_name: str,
    factor_value: Any,
) -> Dict[str, Any]:
    if "target_node" in metrics:
        workload_id = int(metrics["target_node"])
    elif "graph_dataset_index" in metrics:
        workload_id = int(metrics["graph_dataset_index"])
    else:
        raise ValueError(f"Cannot determine workload id from {metric_path}")

    observed_graph = _load_observed_graph(metric_path.parent, manifest["task"], workload_id)
    num_nodes_observed = None
    num_edges_observed = None
    if observed_graph is not None:
        try:
            num_nodes_observed = int(observed_graph.num_nodes)
        except Exception:
            num_nodes_observed = None
        try:
            num_edges_observed = int(observed_graph.edge_index.size(1))
        except Exception:
            num_edges_observed = None

    return {
        "dataset": manifest["dataset"],
        "dataset_slug": manifest["dataset_slug"],
        "task": manifest["task"],
        "model_name": manifest["model_key"],
        "resolved_model_name": manifest["model_name"],
        "method": manifest["method_label"],
        "method_key": manifest["method_key"],
        "factor_name": factor_name or "default",
        "factor_value": factor_value if factor_name else "default",
        "workload_id": workload_id,
        "num_nodes_observed": num_nodes_observed,
        "num_edges_observed": num_edges_observed,
        "candidate_count": int(metrics.get("candidate_count", 0) or 0),
        "verified_witness_count": int(metrics.get("verified_witness_count", metrics.get("num_witnesses", 0)) or 0),
        "selected_witness_count": int(metrics.get("selected_witness_count", 0) or 0),
        "witness_count": int(metrics.get("verified_witness_count", metrics.get("num_witnesses", 0)) or 0),
        "hit_consequent_constraint_count": int(metrics.get("hit_constraint_count", 0) or 0),
        "active_constraint_count": int(metrics.get("active_constraint_count", 0) or 0),
        "covered_constraint_count": int(metrics.get("covered_constraint_count", 0) or 0),
        "coverage_global": float(metrics.get("coverage_ratio_global", 0.0) or 0.0),
        "coverage_normalized": float(metrics.get("coverage_ratio_normalized", 0.0) or 0.0),
        "conciseness": float(metrics.get("avg_conciseness", metrics.get("conciseness", 0.0)) or 0.0),
        "fidelity_minus": float(metrics.get("avg_fidelity_minus", metrics.get("fidelity_minus", 0.0)) or 0.0),
        "runtime_method_only": float(metrics.get("runtime_sec", 0.0) or 0.0),
        "timeout_flag": status_payload.get("status") == "timeout",
        "status": status_payload.get("status", "completed"),
        "run_dir": str(metric_path.parent),
        "metric_file": str(metric_path),
    }


def _method_summary(
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
    factor_name: str,
    factor_value: Any,
    num_workloads: int,
) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "dataset": manifest["dataset"],
            "dataset_slug": manifest["dataset_slug"],
            "task": manifest["task"],
            "model_name": manifest["model_key"],
            "resolved_model_name": manifest["model_name"],
            "method": manifest["method_label"],
            "method_key": manifest["method_key"],
            "factor_name": factor_name or "default",
            "factor_value": factor_value if factor_name else "default",
            "num_workloads": int(num_workloads),
            "num_completed": 0,
            "timeout_count": 1 if status_payload.get("status") == "timeout" else 0,
            "avg_candidate_count": 0.0,
            "avg_verified_witness_count": 0.0,
            "avg_selected_witness_count": 0.0,
            "avg_witness_count": 0.0,
            "avg_hit_consequent_constraint_count": 0.0,
            "avg_active_constraint_count": 0.0,
            "avg_covered_constraint_count": 0.0,
            "avg_coverage_global": 0.0,
            "avg_coverage_normalized": 0.0,
            "avg_conciseness": 0.0,
            "avg_fidelity_minus": 0.0,
            "runtime_total": float(status_payload.get("elapsed_sec", 0.0) or 0.0),
            "runtime_per_workload": 0.0,
            "status": status_payload.get("status", "unknown"),
            "run_dir": status_payload.get("run_dir"),
        }

    return {
        "dataset": manifest["dataset"],
        "dataset_slug": manifest["dataset_slug"],
        "task": manifest["task"],
        "model_name": manifest["model_key"],
        "resolved_model_name": manifest["model_name"],
        "method": manifest["method_label"],
        "method_key": manifest["method_key"],
        "factor_name": factor_name or "default",
        "factor_value": factor_value if factor_name else "default",
        "num_workloads": int(num_workloads),
        "num_completed": int(len(frame)),
        "timeout_count": 1 if status_payload.get("status") == "timeout" else 0,
        "avg_candidate_count": float(frame["candidate_count"].mean()),
        "avg_verified_witness_count": float(frame["verified_witness_count"].mean()),
        "avg_selected_witness_count": float(frame["selected_witness_count"].mean()),
        "avg_witness_count": float(frame["verified_witness_count"].mean()),
        "avg_hit_consequent_constraint_count": float(frame["hit_consequent_constraint_count"].mean()),
        "avg_active_constraint_count": float(frame["active_constraint_count"].mean()),
        "avg_covered_constraint_count": float(frame["covered_constraint_count"].mean()),
        "avg_coverage_global": float(frame["coverage_global"].mean()),
        "avg_coverage_normalized": float(frame["coverage_normalized"].mean()),
        "avg_conciseness": float(frame["conciseness"].mean()),
        "avg_fidelity_minus": float(frame["fidelity_minus"].mean()),
        "runtime_total": float(status_payload.get("elapsed_sec", frame["runtime_method_only"].sum()) or 0.0),
        "runtime_per_workload": float(frame["runtime_method_only"].mean()),
        "status": status_payload.get("status", "completed"),
        "run_dir": status_payload.get("run_dir"),
    }


def _prewarm_observed_graphs(
    dataset_cfg: Dict[str, Any],
    dataset_resource: Any,
    selection: Dict[str, Any],
    constraints: List[Dict[str, Any]],
    run_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    from Run_Experiment import _prepare_observed_graph_workload
    from Run_Experiment_Node import _prepare_observed_node_workload

    workload_records: List[Dict[str, Any]] = []
    if dataset_cfg["task"] == "node":
        data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
        for target_node in selection.get("target_nodes", []):
            observed_subgraph, dropped_edges, node_subset = _prepare_observed_node_workload(
                data,
                int(target_node),
                constraints,
                run_cfg,
            )
            workload_records.append(
                {
                    "workload_id": int(target_node),
                    "num_nodes_observed": int(observed_subgraph.num_nodes),
                    "num_edges_observed": int(observed_subgraph.edge_index.size(1)),
                    "num_dropped_edges": int(len(dropped_edges)),
                    "node_subset_size": int(node_subset.numel()),
                }
            )
        return workload_records

    dataset = dataset_resource["dataset"]
    for pos in selection.get("graph_positions", []):
        _, observed_graph, dropped_edges, dataset_idx, _ = _prepare_observed_graph_workload(
            int(pos),
            dataset_resource,
            dataset,
            constraints,
            run_cfg,
        )
        workload_records.append(
            {
                "workload_id": int(dataset_idx),
                "graph_position": int(pos),
                "num_nodes_observed": int(observed_graph.num_nodes),
                "num_edges_observed": int(observed_graph.edge_index.size(1)),
                "num_dropped_edges": int(len(dropped_edges)),
            }
        )
    return workload_records


def _run_method(
    script_path: Path,
    run_dir: Path,
    run_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    force: bool,
) -> Dict[str, Any]:
    metrics_glob = "metrics_graph_*.json" if dataset_cfg["task"] == "graph" else "metrics_node_*.json"
    status_path = run_dir / "run_status.json"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if not force:
        existing_metrics = _metric_files_for_task(run_dir, dataset_cfg["task"])
        if status_path.exists():
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            if existing_metrics or status_payload.get("status") in {"completed", "timeout", "failed"}:
                status_payload["run_dir"] = str(run_dir)
                return status_payload
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(run_cfg, handle, sort_keys=False)

    manifest = {
        "experiment_name": run_cfg.get("exp_name", "bundle_run"),
        "experiment_group": "bundle",
        "dataset": run_cfg["data_name"],
        "dataset_slug": run_cfg["dataset_slug"],
        "task": dataset_cfg["task"],
        "model_key": run_cfg["model_key"],
        "model_name": run_cfg["model_name"],
        "method_key": run_cfg["method_key"],
        "method_label": run_cfg["method_label"],
        "seed": int(run_cfg["random_seed"]),
        "K": int(run_cfg["K"]),
        "k": int(run_cfg["local_budget_k"]),
        "L": int(run_cfg["L"]),
        "sigma_size": int(run_cfg["sigma_size"]),
        "incompleteness": float(run_cfg["incompleteness"]),
        "target_ratio": float(run_cfg.get("target_ratio", 1.0)),
        "gamma": float(run_cfg.get("gamma", 1.0)),
        "alpha": float(run_cfg.get("alpha", 1.0)),
        "beta": float(run_cfg.get("beta", 1.0)),
        "constraint_source": run_cfg.get("constraint_source", "static"),
        "config_path": str(config_path),
        "target_nodes": run_cfg.get("target_nodes", []),
        "graph_positions": run_cfg.get("graph_positions", []),
    }
    write_run_manifest(run_dir / "run_manifest.json", manifest)

    timeout_sec = None
    timeout_key = f"{run_cfg['method_key']}_timeout_sec"
    raw_timeout = None
    if run_cfg.get(timeout_key) is not None:
        raw_timeout = run_cfg.get(timeout_key)
    elif run_cfg.get("run_timeout_sec") is not None:
        raw_timeout = run_cfg.get("run_timeout_sec")
    if raw_timeout not in (None, "", False):
        parsed_timeout = float(raw_timeout)
        if parsed_timeout > 0:
            timeout_sec = parsed_timeout

    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    if dataset_cfg["task"] == "graph" and len(run_cfg.get("graph_positions", [])) == 1:
        cmd += ["--input", str(run_cfg["graph_positions"][0])]
    elif run_cfg.get("run_all", False):
        cmd.append("--run_all")
    elif dataset_cfg["task"] == "graph":
        cmd += ["--input", str(run_cfg.get("graph_positions", [0])[0])]
    else:
        cmd += ["--input", "0"]

    t_start = time.time()
    write_run_status(status_path, {"status": "running", "started_at": t_start, "timeout_sec": timeout_sec})
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=timeout_sec)
        status = {
            "status": "completed",
            "started_at": t_start,
            "finished_at": time.time(),
            "elapsed_sec": time.time() - t_start,
            "timeout_sec": timeout_sec,
            "run_dir": str(run_dir),
        }
    except subprocess.TimeoutExpired:
        status = {
            "status": "timeout",
            "started_at": t_start,
            "finished_at": time.time(),
            "elapsed_sec": time.time() - t_start,
            "timeout_sec": timeout_sec,
            "run_dir": str(run_dir),
        }
    except subprocess.CalledProcessError as exc:
        status = {
            "status": "failed",
            "started_at": t_start,
            "finished_at": time.time(),
            "elapsed_sec": time.time() - t_start,
            "timeout_sec": timeout_sec,
            "returncode": exc.returncode,
            "run_dir": str(run_dir),
        }
    write_run_status(status_path, status)
    return status


def main() -> None:
    from constraint_mining import resolve_constraints
    from utils import dataset_func, set_seed

    args = parse_args()
    bundle_path = Path(args.config)
    bundle_cfg = load_yaml(bundle_path)
    default_cfg = load_yaml(Path(bundle_cfg.get("default_config_path", ROOT / "configs" / "local" / "default.yaml")))
    dataset_cfg_dir = Path(bundle_cfg.get("dataset_config_dir", ROOT / "configs" / "local"))

    dataset_key = str(bundle_cfg["dataset"]).lower()
    dataset_cfg = load_yaml(dataset_cfg_dir / f"{dataset_key}.yaml")
    dataset_cfg["slug"] = dataset_key

    model_key = str(bundle_cfg["model_name"])
    if model_key not in dataset_cfg["models"]:
        raise ValueError(f"Unknown model_name '{model_key}' for dataset {dataset_cfg['data_name']}")

    methods = [_normalize_method_key(name) for name in bundle_cfg["methods"]]
    run_name = str(bundle_cfg["run_name"])
    output_root = Path(bundle_cfg.get("output_root", ROOT / "outputs" / "raw" / "bundles"))
    reuse_observed_graph_cache = bool(bundle_cfg.get("reuse_observed_graph_cache", True))
    base_overrides = _bundle_base_overrides(bundle_cfg)
    factor_pairs = _coerce_factor_values(bundle_cfg)

    set_seed(int(default_cfg.get("defaults", {}).get("random_seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_cfg = {"name": run_name, "group": "bundle", "base": base_overrides}
    seed_run_cfg = build_run_config(default_cfg, dataset_cfg, seed_cfg, methods[0], model_key, {})
    dataset_resource = dataset_func(seed_run_cfg)

    per_workload_rows: List[Dict[str, Any]] = []
    method_summary_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "run_name": run_name,
        "bundle_config_path": str(bundle_path),
        "bundle_config": bundle_cfg,
        "dataset": dataset_cfg["data_name"],
        "dataset_slug": dataset_key,
        "model_name": model_key,
        "resolved_model_name": dataset_cfg["models"][model_key],
        "methods": methods,
        "reuse_observed_graph_cache": reuse_observed_graph_cache,
        "started_at": time.time(),
        "factors": [],
    }

    csv_root = ROOT / "outputs" / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)
    per_workload_path = csv_root / f"{run_name}_per_workload.csv"
    method_summary_path = csv_root / f"{run_name}_method_summary.csv"
    metadata_path = csv_root / f"{run_name}_metadata.json"
    if args.force or bool(bundle_cfg.get("force", False)):
        for stale_path in (per_workload_path, method_summary_path, metadata_path):
            if stale_path.exists():
                stale_path.unlink()
    script_path = resolved_entry_script(dataset_cfg, {"runner_script": None})

    predicted_type_sources = {"predicted", "predicted_label", "model_prediction"}
    shared_model: Optional[torch.nn.Module] = None
    if str(seed_run_cfg.get("constraint_type_source", "")).lower() in predicted_type_sources:
        shared_model = _load_model_checkpoint(seed_run_cfg, device)

    for factor_name, factor_value in factor_pairs:
        combo = {}
        if factor_name:
            combo[factor_name] = factor_value
        exp_cfg = {"name": run_name, "group": "bundle", "base": base_overrides}
        factor_run_cfg = build_run_config(default_cfg, dataset_cfg, exp_cfg, methods[0], model_key, combo)
        selection = select_targets(dataset_cfg, dataset_resource, factor_run_cfg)
        factor_run_cfg.update(selection)
        factor_run_cfg["run_all"] = True

        constraints = resolve_constraints(
            factor_run_cfg,
            dataset_resource,
            model=shared_model,
            device=device,
            save_dir=None,
        )

        if not reuse_observed_graph_cache:
            _clear_observed_cache(dataset_cfg["data_name"])
        workload_records = _prewarm_observed_graphs(dataset_cfg, dataset_resource, selection, constraints, factor_run_cfg)

        factor_meta = {
            "factor_name": factor_name,
            "factor_value": factor_value if factor_name else "default",
            "num_workloads": len(workload_records),
            "workload_ids": [int(row["workload_id"]) for row in workload_records],
            "resolved_constraint_count": int(len(constraints)),
            "constraint_names": [str(tgd.get("name", "")) for tgd in constraints],
            "observed_cache_reused": bool(reuse_observed_graph_cache),
            "prewarmed_workloads": workload_records,
            "methods": [],
        }

        for method_key in methods:
            run_cfg = build_run_config(default_cfg, dataset_cfg, exp_cfg, method_key, model_key, combo)
            run_cfg.update(selection)
            run_cfg["run_all"] = True
            run_dir = _bundle_run_dir(output_root, run_name, dataset_key, model_key, factor_name, factor_value, method_key)
            run_cfg["save_dir"] = str(run_dir)
            status = _run_method(script_path, run_dir, run_cfg, dataset_cfg, force=args.force or bool(bundle_cfg.get("force", False)))

            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            metric_rows: List[Dict[str, Any]] = []
            for metric_path in _metric_files_for_task(run_dir, dataset_cfg["task"]):
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                row = _per_workload_row(metric_path, metrics, manifest, status, factor_name, factor_value)
                metric_rows.append(row)
                per_workload_rows.append(row)

            summary = _method_summary(metric_rows, manifest, status, factor_name, factor_value, len(workload_records))
            method_summary_rows.append(summary)
            factor_meta["methods"].append(summary)

            pd.DataFrame(per_workload_rows).to_csv(per_workload_path, index=False)
            pd.DataFrame(method_summary_rows).to_csv(method_summary_path, index=False)

        metadata["factors"].append(factor_meta)
        metadata["updated_at"] = time.time()
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    metadata["finished_at"] = time.time()
    metadata["duration_sec"] = metadata["finished_at"] - metadata["started_at"]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Per-workload CSV: {csv_root / f'{run_name}_per_workload.csv'}")
    print(f"Method summary CSV: {csv_root / f'{run_name}_method_summary.csv'}")
    print(f"Metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()
