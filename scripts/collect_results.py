#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "outputs" / "raw"
CSV_ROOT = ROOT / "outputs" / "csv"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def witness_average(metrics: Dict[str, Any], key: str) -> float | None:
    witnesses = metrics.get("witnesses", [])
    if not witnesses:
        return None
    values = [w.get(key) for w in witnesses if w.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def collect_records() -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    target_records: List[Dict[str, Any]] = []
    witness_records: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []

    for manifest_path in RAW_ROOT.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        status_path = run_dir / "run_status.json"
        status_payload = read_json(status_path) if status_path.exists() else {}
        resolved_path = run_dir / "resolved_constraints.json"
        resolved = read_json(resolved_path) if resolved_path.exists() else {}
        sigma_resolved = int(resolved.get("num_constraints", 0))

        metric_files = sorted(run_dir.glob("metrics_graph_*.json")) + sorted(run_dir.glob("metrics_node_*.json"))
        if not metric_files:
            if status_payload.get("status") in {"timeout", "failed"}:
                run_records.append(
                    {
                        "experiment_name": manifest["experiment_name"],
                        "experiment_group": manifest["experiment_group"],
                        "dataset": manifest["dataset"],
                        "dataset_slug": manifest["dataset_slug"],
                        "task": manifest["task"],
                        "model_key": manifest["model_key"],
                        "model": manifest["model_name"],
                        "method": manifest["method_label"],
                        "method_key": manifest["method_key"],
                        "seed": manifest["seed"],
                        "K": manifest["K"],
                        "k": manifest["k"],
                        "L": manifest["L"],
                        "sigma_size": manifest["sigma_size"],
                        "sigma_requested": manifest["sigma_size"],
                        "sigma_resolved": sigma_resolved,
                        "incompleteness": manifest["incompleteness"],
                        "target_ratio": manifest["target_ratio"],
                        "gamma": manifest.get("gamma"),
                        "alpha": manifest.get("alpha"),
                        "beta": manifest.get("beta"),
                        "constraint_source": manifest.get("constraint_source", "static"),
                        "runtime": None,
                        "runtime_sec": None,
                        "runtime_total": status_payload.get("elapsed_sec"),
                        "runtime_per_workload": None,
                        "conciseness": None,
                        "alignment": None,
                        "coverage": None,
                        "coverage_global": None,
                        "coverage_normalized": None,
                        "avg_hit_constraint_count": None,
                        "avg_active_constraint_count": None,
                        "avg_covered_constraint_count": None,
                        "fidelity_minus": None,
                        "one_minus_fidelity_minus": None,
                        "F_set": None,
                        "num_targets": 0,
                        "avg_candidate_count": None,
                        "avg_verified_witness_count": None,
                        "avg_selected_witness_count": None,
                        "avg_num_witnesses": None,
                        "run_dir": str(run_dir),
                        "status": status_payload.get("status"),
                        "timeout_sec": status_payload.get("timeout_sec"),
                    }
                )
            continue

        run_level_rows = []
        for metric_path in metric_files:
            metrics = read_json(metric_path)
            fidelity_minus = metrics.get("avg_fidelity_minus", metrics.get("fidelity_minus"))
            conciseness = metrics.get("avg_conciseness", metrics.get("conciseness"))
            coverage = metrics.get("coverage_ratio")
            coverage_global = metrics.get("coverage_ratio_global", coverage)
            coverage_normalized = metrics.get("coverage_ratio_normalized", coverage)
            hit_constraint_count = metrics.get("hit_constraint_count")
            active_constraint_count = metrics.get("active_constraint_count")
            covered_constraint_count = metrics.get("covered_constraint_count", metrics.get("coverage_size"))
            runtime = metrics.get("runtime_sec", metrics.get("elapsed_time", metrics.get("total_time")))
            alignment = witness_average(metrics, "alignment")
            q_avg = witness_average(metrics, "q")

            base_row = {
                "experiment_name": manifest["experiment_name"],
                "experiment_group": manifest["experiment_group"],
                "dataset": manifest["dataset"],
                "dataset_slug": manifest["dataset_slug"],
                "task": manifest["task"],
                "model_key": manifest["model_key"],
                "model": manifest["model_name"],
                "method": manifest["method_label"],
                "method_key": manifest["method_key"],
                "seed": manifest["seed"],
                "K": manifest["K"],
                "k": manifest["k"],
                "L": manifest["L"],
                "sigma_size": manifest["sigma_size"],
                "sigma_requested": manifest["sigma_size"],
                "sigma_resolved": sigma_resolved,
                "incompleteness": manifest["incompleteness"],
                "target_ratio": manifest["target_ratio"],
                "gamma": manifest.get("gamma"),
                "alpha": manifest.get("alpha"),
                "beta": manifest.get("beta"),
                "constraint_source": manifest.get("constraint_source", "static"),
                "runtime": runtime,
                "runtime_sec": runtime,
                "conciseness": conciseness,
                "alignment": alignment,
                "coverage": coverage,
                "coverage_global": coverage_global,
                "coverage_normalized": coverage_normalized,
                "hit_constraint_count": hit_constraint_count,
                "active_constraint_count": active_constraint_count,
                "covered_constraint_count": covered_constraint_count,
                "fidelity_minus": fidelity_minus,
                "one_minus_fidelity_minus": (None if fidelity_minus is None else 1.0 - float(fidelity_minus)),
                "F_set": metrics.get("set_objective_F"),
                "candidate_count": metrics.get("candidate_count", metrics.get("distinct_candidates_generated")),
                "verified_witness_count": metrics.get("verified_witness_count", metrics.get("num_witnesses")),
                "selected_witness_count": metrics.get("selected_witness_count"),
                "num_witnesses": metrics.get("verified_witness_count", metrics.get("num_witnesses")),
                "distinct_candidates_generated": metrics.get("distinct_candidates_generated"),
                "num_candidates_verified": metrics.get("num_candidates_verified"),
                "num_candidates_admitted": metrics.get("num_candidates_admitted"),
                "fallback_used": metrics.get("fallback_used"),
                "fallback_selected": metrics.get("fallback_selected"),
                "covered_constraints": "|".join(metrics.get("covered_constraints", [])),
                "run_dir": str(run_dir),
                "metric_file": str(metric_path),
                "status": status_payload.get("status", "completed"),
                "timeout_sec": status_payload.get("timeout_sec"),
            }
            if "target_node" in metrics:
                base_row["target_id"] = int(metrics["target_node"])
            elif "graph_dataset_index" in metrics:
                base_row["target_id"] = int(metrics["graph_dataset_index"])
            else:
                base_row["target_id"] = None
            target_records.append(base_row)
            run_level_rows.append(base_row)

            for witness in metrics.get("witnesses", []):
                witness_records.append(
                    {
                        **base_row,
                        "witness_index": witness.get("index"),
                        "witness_num_nodes": witness.get("num_nodes"),
                        "witness_num_edges": witness.get("num_edges"),
                        "witness_conc": witness.get("conc"),
                        "witness_conciseness": witness.get("conciseness"),
                        "witness_alignment": witness.get("alignment"),
                        "witness_q": witness.get("q"),
                        "witness_fidelity_minus": witness.get("fidelity_minus"),
                        "witness_delta_edge_count": len(witness.get("delta_edges", [])),
                        "witness_supporting_edge_count": len(witness.get("supporting_edges", [])),
                        "witness_grounded_constraints": "|".join(witness.get("grounded_constraints", [])),
                    }
                )

        if run_level_rows:
            frame = pd.DataFrame(run_level_rows)
            runtime_per_workload = float(frame["runtime"].mean())
            runtime_total = status_payload.get("elapsed_sec")
            if runtime_total is None:
                runtime_total = float(frame["runtime"].sum())
            run_records.append(
                {
                    "experiment_name": manifest["experiment_name"],
                    "experiment_group": manifest["experiment_group"],
                    "dataset": manifest["dataset"],
                    "dataset_slug": manifest["dataset_slug"],
                    "task": manifest["task"],
                    "model_key": manifest["model_key"],
                    "model": manifest["model_name"],
                    "method": manifest["method_label"],
                    "method_key": manifest["method_key"],
                    "seed": manifest["seed"],
                    "K": manifest["K"],
                    "k": manifest["k"],
                    "L": manifest["L"],
                    "sigma_size": manifest["sigma_size"],
                    "sigma_requested": manifest["sigma_size"],
                    "sigma_resolved": sigma_resolved,
                    "incompleteness": manifest["incompleteness"],
                    "target_ratio": manifest["target_ratio"],
                    "gamma": manifest.get("gamma"),
                    "alpha": manifest.get("alpha"),
                    "beta": manifest.get("beta"),
                    "constraint_source": manifest.get("constraint_source", "static"),
                    # Backward-compatible aliases keep the old per-workload semantics.
                    "runtime": runtime_per_workload,
                    "runtime_sec": runtime_per_workload,
                    "runtime_total": float(runtime_total),
                    "runtime_per_workload": runtime_per_workload,
                    "conciseness": float(frame["conciseness"].mean()),
                    "alignment": float(frame["alignment"].dropna().mean()) if frame["alignment"].notna().any() else None,
                    "coverage": float(frame["coverage"].mean()),
                    "coverage_global": float(frame["coverage_global"].mean()) if frame["coverage_global"].notna().any() else None,
                    "coverage_normalized": float(frame["coverage_normalized"].mean()) if frame["coverage_normalized"].notna().any() else None,
                    "avg_hit_constraint_count": float(frame["hit_constraint_count"].dropna().mean()) if frame["hit_constraint_count"].notna().any() else None,
                    "avg_active_constraint_count": float(frame["active_constraint_count"].dropna().mean()) if frame["active_constraint_count"].notna().any() else None,
                    "avg_covered_constraint_count": float(frame["covered_constraint_count"].dropna().mean()) if frame["covered_constraint_count"].notna().any() else None,
                    "fidelity_minus": float(frame["fidelity_minus"].mean()),
                    "one_minus_fidelity_minus": float(frame["one_minus_fidelity_minus"].mean()),
                    "F_set": float(frame["F_set"].dropna().mean()) if frame["F_set"].notna().any() else None,
                    "num_targets": int(len(frame)),
                    "avg_candidate_count": float(frame["candidate_count"].dropna().mean()) if frame["candidate_count"].notna().any() else None,
                    "avg_verified_witness_count": float(frame["verified_witness_count"].dropna().mean()) if frame["verified_witness_count"].notna().any() else None,
                    "avg_selected_witness_count": float(frame["selected_witness_count"].dropna().mean()) if frame["selected_witness_count"].notna().any() else None,
                    "avg_num_witnesses": float(frame["num_witnesses"].dropna().mean()) if frame["num_witnesses"].notna().any() else None,
                    "distinct_candidates_generated": float(frame["distinct_candidates_generated"].dropna().mean()) if frame["distinct_candidates_generated"].notna().any() else None,
                    "num_candidates_verified": float(frame["num_candidates_verified"].dropna().mean()) if frame["num_candidates_verified"].notna().any() else None,
                    "num_candidates_admitted": float(frame["num_candidates_admitted"].dropna().mean()) if frame["num_candidates_admitted"].notna().any() else None,
                    "fallback_used_ratio": _bool_ratio(frame["fallback_used"]) if "fallback_used" in frame else None,
                    "fallback_selected_ratio": _bool_ratio(frame["fallback_selected"]) if "fallback_selected" in frame else None,
                    "run_dir": str(run_dir),
                    "status": status_payload.get("status", "completed"),
                    "timeout_sec": status_payload.get("timeout_sec"),
                }
            )

    return target_records, witness_records, run_records


def _bool_ratio(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = [bool(v) for v in series.tolist() if v is not None and not pd.isna(v)]
    if not values:
        return None
    return float(sum(1.0 for v in values if v) / len(values))


def main() -> None:
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    target_records, witness_records, run_records = collect_records()

    target_df = pd.DataFrame(target_records)
    witness_df = pd.DataFrame(witness_records)
    run_df = pd.DataFrame(run_records)

    target_df.to_csv(CSV_ROOT / "local_target_metrics.csv", index=False)
    witness_df.to_csv(CSV_ROOT / "local_witness_metrics.csv", index=False)
    run_df.to_csv(CSV_ROOT / "local_run_metrics.csv", index=False)

    legacy_csv = CSV_ROOT / "legacy_overall_reference.csv"
    if legacy_csv.exists():
        legacy_df = pd.read_csv(legacy_csv)
        merged_df = pd.concat([legacy_df, run_df], ignore_index=True, sort=False)
        merged_df.to_csv(CSV_ROOT / "local_run_metrics_merged.csv", index=False)
        print(f"Saved: {CSV_ROOT / 'local_run_metrics_merged.csv'}")

    print(f"Saved: {CSV_ROOT / 'local_target_metrics.csv'}")
    print(f"Saved: {CSV_ROOT / 'local_witness_metrics.csv'}")
    print(f"Saved: {CSV_ROOT / 'local_run_metrics.csv'}")


if __name__ == "__main__":
    main()
