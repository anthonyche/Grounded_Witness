#!/usr/bin/env python3
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_local_experiment import build_run_config, load_yaml  # type: ignore
from constraint_mining import resolve_constraints
from model import get_model
from Run_Experiment_Node import (
    _load_trained_model,
    _prepare_observed_node_workload,
    _run_one_node_apxchase,
)
from apxchase import ApxChase
from heuchase import HeuChase
from exhaustchase import ExhaustChase
from utils import dataset_func, set_seed


def _load_constraint_names(csv_path: Path) -> List[str]:
    frame = pd.read_csv(csv_path)
    if "selected" in frame.columns:
        frame = frame[frame["selected"].fillna(False)]
    return [str(name) for name in frame["constraint_name"].tolist()]


def _filter_constraints_by_name(constraints: List[dict], names: Iterable[str]) -> List[dict]:
    wanted = list(names)
    by_name = {str(tgd.get("name", "")): tgd for tgd in constraints}
    filtered: List[dict] = []
    for name in wanted:
        if name not in by_name:
            raise KeyError(f"Constraint '{name}' not found in mined DBLP candidate pool.")
        filtered.append(by_name[name])
    return filtered


def _read_metrics(save_dir: Path, target_node: int) -> Dict[str, Any]:
    metrics_path = save_dir / f"metrics_node_{int(target_node)}.json"
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _method_label(method_key: str) -> str:
    return {
        "apxchase": "ApxC",
        "heuchase": "HeuC",
        "exhaustchase": "Exh",
    }[method_key]


def _instantiate_chaser(method_key: str, model: torch.nn.Module, constraints: List[dict], cfg: Dict[str, Any]):
    common = dict(
        model=model,
        Sigma=constraints,
        L=int(cfg.get("L", 2)),
        k=int(cfg.get("K", cfg.get("k", 6))),
        B=int(cfg.get("local_budget_k", 2)),
        alpha=float(cfg.get("alpha", 0.5)),
        beta=float(cfg.get("beta", 0.5)),
        gamma=float(cfg.get("gamma", 1.0)),
        seed_per_constraint=int(cfg.get("apx_seed_per_constraint", 2)),
        candidate_expand_steps=int(cfg.get("apx_candidate_expand_steps", 2)),
        candidate_branch_factor=int(cfg.get("apx_candidate_branch_factor", 3)),
        candidate_beam_width=int(cfg.get("apx_candidate_beam_width", 6)),
        candidate_max_masks=int(cfg.get("apx_candidate_max_masks", 48)),
        legacy_prefix_checkpoints=int(cfg.get("apx_legacy_prefix_checkpoints", 6)),
        use_ranked_candidate_prioritization=bool(cfg.get("apx_use_ranked_candidate_prioritization", True)),
        use_task_aware_hybrid_generation=bool(cfg.get("apx_use_task_aware_hybrid_generation", False)),
        ranking_pool_factor=int(cfg.get("apx_ranking_pool_factor", 3)),
        ranking_diversity_bonus=float(cfg.get("apx_ranking_diversity_bonus", 0.2)),
        max_near_full_candidates=int(cfg.get("apx_max_near_full_candidates", 16)),
        near_full_delete_budget=int(cfg.get("apx_near_full_delete_budget", 3)),
        near_full_branch_factor=int(cfg.get("apx_near_full_branch_factor", 6)),
        near_full_beam_width=int(cfg.get("apx_near_full_beam_width", 4)),
        debug=False,
    )
    if method_key == "apxchase":
        return ApxChase(**common)
    if method_key == "heuchase":
        return HeuChase(
            **common,
            m=int(cfg.get("heuchase_m", 20)),
            noise_std=float(cfg.get("heuchase_noise_std", 0.2)),
        )
    if method_key == "exhaustchase":
        return ExhaustChase(
            **common,
            max_enforce_iterations=int(cfg.get("max_enforce_iterations", 100)),
        )
    raise ValueError(f"Unsupported method_key: {method_key}")


def _build_summary(frame: pd.DataFrame, variant: str, method: str, candidate_pool_size: int, final_selected: int) -> Dict[str, Any]:
    if frame.empty:
        return {
            "variant": variant,
            "method": method,
            "candidate_pool_size": candidate_pool_size,
            "final_selected_constraint_count": final_selected,
            "num_workloads": 0,
            "num_completed": 0,
            "timeout_count": 0,
            "avg_hit_consequent_constraint_count": 0.0,
            "avg_active_constraint_count": 0.0,
            "avg_covered_constraint_count": 0.0,
            "avg_coverage_global": 0.0,
            "avg_coverage_normalized": 0.0,
            "nonzero_coverage_workload_ratio": 0.0,
            "avg_conciseness": 0.0,
            "avg_fidelity_minus": 0.0,
            "runtime_total": 0.0,
        }
    return {
        "variant": variant,
        "method": method,
        "candidate_pool_size": candidate_pool_size,
        "final_selected_constraint_count": final_selected,
        "num_workloads": int(len(frame)),
        "num_completed": int((frame["status"] == "completed").sum()),
        "timeout_count": int(frame["timeout_flag"].sum()),
        "avg_hit_consequent_constraint_count": float(frame["hit_consequent_constraint_count"].mean()),
        "avg_active_constraint_count": float(frame["active_constraint_count"].mean()),
        "avg_covered_constraint_count": float(frame["covered_constraint_count"].mean()),
        "avg_coverage_global": float(frame["coverage_global"].mean()),
        "avg_coverage_normalized": float(frame["coverage_normalized"].mean()),
        "nonzero_coverage_workload_ratio": float((frame["coverage_normalized"] > 0).mean()),
        "avg_conciseness": float(frame["conciseness"].mean()),
        "avg_fidelity_minus": float(frame["fidelity_minus"].mean()),
        "runtime_total": float(frame["runtime_method_only"].sum()),
    }


class _PerWorkloadTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _PerWorkloadTimeout()


def _run_with_timeout(timeout_sec: int | None, fn, *args, **kwargs):
    if not timeout_sec:
        return fn(*args, **kwargs)
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _run_one_workload_subprocess_worker(
    queue: mp.Queue,
    method_key: str,
    method_cfg: Dict[str, Any],
    constraints: List[dict],
    target_node: int,
    observed_subgraph: Any,
    dropped_edges: List[Tuple[int, int]],
) -> None:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_trained_model(method_cfg, device)
        chaser = _instantiate_chaser(method_key, model, constraints, method_cfg)
        elapsed, witness_count, fidelity_minus, conciseness, coverage = _run_one_node_apxchase(
            int(target_node),
            observed_subgraph,
            dropped_edges,
            constraints,
            method_cfg,
            device,
            chaser,
        )
        queue.put(
            {
                "elapsed": elapsed,
                "witness_count": witness_count,
                "fidelity_minus": fidelity_minus,
                "conciseness": conciseness,
                "coverage": coverage,
            }
        )
    except Exception as exc:
        queue.put({"error": repr(exc)})


def _run_one_workload_subprocess(
    method_key: str,
    method_cfg: Dict[str, Any],
    constraints: List[dict],
    target_node: int,
    observed_subgraph: Any,
    dropped_edges: List[Tuple[int, int]],
    timeout_sec: int,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_one_workload_subprocess_worker,
        args=(
            queue,
            method_key,
            method_cfg,
            constraints,
            int(target_node),
            observed_subgraph,
            list(dropped_edges),
        ),
    )
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "timeout", "elapsed": float(timeout_sec)}
    if proc.exitcode not in (0, None):
        payload = queue.get() if not queue.empty() else {"error": f"child_exit_{proc.exitcode}"}
        payload["status"] = "failed"
        return payload
    payload = queue.get() if not queue.empty() else {}
    payload["status"] = "completed"
    return payload


def main() -> None:
    default_cfg = load_yaml(ROOT / "configs" / "local" / "default.yaml")
    dataset_cfg = load_yaml(ROOT / "configs" / "local" / "dblp.yaml")
    dataset_cfg["slug"] = "dblp"
    bundle_cfg = load_yaml(ROOT / "config.yaml")
    target_nodes = [int(x) for x in bundle_cfg["target_nodes"]]

    run_spec = {
        "name": "dblp_constraint_three_way_all_methods",
        "group": "analysis",
        "base": {
            "target_nodes": target_nodes,
            "run_all": True,
            "target_ratio": 0.01,
            "sigma_size": 20,
            "constraint_source": "mined",
            "constraint_type_source": "dblp_native_bucket",
            "constraint_rule_mode": "standard_backchase",
            "constraint_filter_target_matchability": True,
            "constraint_target_probe_samples": 64,
            "constraint_target_min_hit": 1,
            "constraint_use_cache": False,
            "preserve_connectivity": True,
            "max_masks": 1,
            "L": 2,
            "local_budget_k": 2,
            "K": 6,
            "random_seed": 42,
            "gamma": 1.0,
            "alpha": 0.5,
            "beta": 0.5,
            "incompleteness": 0.05,
        },
    }
    run_cfg = build_run_config(default_cfg, dataset_cfg, run_spec, "apxchase", "gcn2", {})
    run_cfg["target_nodes"] = target_nodes
    run_cfg["run_all"] = True

    set_seed(int(run_cfg.get("random_seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_resource = dataset_func(run_cfg)
    data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
    model = _load_trained_model(run_cfg, device)

    # Shared observed graphs come from one fixed 64-rule candidate pool so all
    # three selected pools are compared on exactly the same workloads/graphs.
    observed_cfg = dict(run_cfg)
    observed_cfg["sigma_size"] = 64
    observed_cfg["constraint_limit"] = 64
    observed_cfg["constraint_max_patterns"] = 64
    full_candidate_constraints = resolve_constraints(observed_cfg, dataset_resource, model=model, device=device, save_dir=None)
    candidate_pool_size = len(full_candidate_constraints)

    original_constraints = resolve_constraints(run_cfg, dataset_resource, model=model, device=device, save_dir=None)
    original_names = [str(tgd.get("name", "")) for tgd in original_constraints]
    coverage_only_names = _load_constraint_names(ROOT / "outputs" / "csv" / "dblp_constraint_filtered_pool.csv")
    balanced_names = _load_constraint_names(ROOT / "outputs" / "csv" / "dblp_constraint_balanced_pool.csv")

    pools = {
        "original": original_names,
        "coverage_only": coverage_only_names,
        "balanced_p7_t0p4": balanced_names,
    }

    observed_workloads: Dict[int, Tuple[torch.Tensor, List[Tuple[int, int]]]] = {}
    shared_rows: List[Dict[str, Any]] = []
    for target_node in target_nodes:
        observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, int(target_node), full_candidate_constraints, observed_cfg)
        observed_workloads[int(target_node)] = (observed_subgraph.cpu(), list(dropped_edges))
        shared_rows.append(
            {
                "workload_id": int(target_node),
                "num_nodes_observed": int(observed_subgraph.num_nodes),
                "num_edges_observed": int(observed_subgraph.edge_index.size(1)),
                "num_dropped_edges": int(len(dropped_edges)),
                "shared_observed_source": "full_candidate_pool_64",
            }
        )

    per_workload_rows: List[Dict[str, Any]] = []
    method_summary_rows: List[Dict[str, Any]] = []
    raw_root = ROOT / "outputs" / "raw" / "analysis" / "dblp_constraint_three_way_all_methods"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    per_workload_csv = ROOT / "outputs" / "csv" / "dblp_constraint_three_way_all_methods_per_workload.csv"
    summary_csv = ROOT / "outputs" / "csv" / "dblp_constraint_three_way_all_methods_summary.csv"
    if per_workload_csv.exists():
        per_workload_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()
    exhaust_timeout_sec = 45

    for variant, names in pools.items():
        selected_constraints = _filter_constraints_by_name(full_candidate_constraints, names)
        for method_key in ("apxchase", "heuchase", "exhaustchase"):
            method_label = _method_label(method_key)
            print(f"[constraint-pools] variant={variant} method={method_label} workloads={len(target_nodes)}")
            method_cfg = dict(run_cfg)
            save_dir = raw_root / variant / method_key
            save_dir.mkdir(parents=True, exist_ok=True)
            method_cfg["save_dir"] = str(save_dir)

            chaser = _instantiate_chaser(method_key, model, selected_constraints, method_cfg)

            for index, target_node in enumerate(target_nodes, start=1):
                print(f"  - workload {index}/{len(target_nodes)} target={int(target_node)}")
                observed_subgraph, dropped_edges = observed_workloads[int(target_node)]
                elapsed = 0.0
                witness_count = 0
                fidelity_minus = 0.0
                conciseness = 0.0
                coverage = 0.0
                status = "completed"
                timeout_flag = False
                timeout_sec = exhaust_timeout_sec if method_key == "exhaustchase" else None
                try:
                    if method_key == "exhaustchase":
                        payload = _run_one_workload_subprocess(
                            method_key,
                            method_cfg,
                            selected_constraints,
                            int(target_node),
                            observed_subgraph.clone(),
                            list(dropped_edges),
                            int(timeout_sec or 0),
                        )
                        status = str(payload.get("status", "failed"))
                        if status == "completed":
                            elapsed = float(payload.get("elapsed", 0.0))
                            witness_count = int(payload.get("witness_count", 0))
                            fidelity_minus = float(payload.get("fidelity_minus", 0.0))
                            conciseness = float(payload.get("conciseness", 0.0))
                            coverage = float(payload.get("coverage", 0.0))
                            metrics = _read_metrics(save_dir, int(target_node))
                        else:
                            timeout_flag = status == "timeout"
                            elapsed = float(payload.get("elapsed", float(timeout_sec or 0.0)))
                            metrics = {}
                            print(f"    {status} after {elapsed:.1f}s")
                    else:
                        elapsed, witness_count, fidelity_minus, conciseness, coverage = _run_one_node_apxchase(
                            int(target_node),
                            observed_subgraph.clone(),
                            list(dropped_edges),
                            selected_constraints,
                            method_cfg,
                            device,
                            chaser,
                        )
                        metrics = _read_metrics(save_dir, int(target_node))
                except Exception as exc:
                    status = "failed"
                    metrics = {}
                    print(f"    failed: {exc}")
                per_workload_rows.append(
                    {
                        "variant": variant,
                        "method": method_label,
                        "method_key": method_key,
                        "candidate_pool_size": candidate_pool_size,
                        "final_selected_constraint_count": len(selected_constraints),
                        "workload_id": int(target_node),
                        "num_nodes_observed": int(observed_subgraph.num_nodes),
                        "num_edges_observed": int(observed_subgraph.edge_index.size(1)),
                        "witness_count": int(metrics.get("num_witnesses", witness_count)),
                        "hit_consequent_constraint_count": int(metrics.get("hit_constraint_count", 0)),
                        "active_constraint_count": int(metrics.get("active_constraint_count", 0)),
                        "covered_constraint_count": int(metrics.get("covered_constraint_count", 0)),
                        "coverage_global": float(metrics.get("coverage_ratio_global", 0.0)),
                        "coverage_normalized": float(metrics.get("coverage_ratio_normalized", coverage)),
                        "conciseness": float(metrics.get("avg_conciseness", conciseness)),
                        "fidelity_minus": float(metrics.get("avg_fidelity_minus", fidelity_minus)),
                        "runtime_method_only": float(metrics.get("runtime_sec", elapsed)),
                        "timeout_flag": timeout_flag,
                        "status": status,
                    }
                )
                pd.DataFrame(per_workload_rows).to_csv(per_workload_csv, index=False)

            frame = pd.DataFrame(
                [row for row in per_workload_rows if row["variant"] == variant and row["method"] == method_label]
            )
            method_summary_rows.append(
                _build_summary(frame, variant, method_label, candidate_pool_size, len(selected_constraints))
            )

            pd.DataFrame(method_summary_rows).to_csv(summary_csv, index=False)

    pd.DataFrame(shared_rows).to_csv(
        ROOT / "outputs" / "csv" / "dblp_constraint_three_way_shared_observed.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
