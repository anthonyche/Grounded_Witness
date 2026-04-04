"""
Run_Experiment_Node.py
----------------------
Node classification explanation pipeline for Cora, CiteSeer, PubMed, etc.

Similar to Run_Experiment.py but adapted for node-level tasks:
1. Load pre-trained node classification GNN
2. Select target nodes from test set
3. Extract L-hop subgraphs and apply constraint-driven edge masking
4. Run ApxChase/HeuChase/ExhaustChase/GNNExplainer/PGExplainer
5. Compute metrics (Fidelity-, Conciseness, Coverage) and save results

Usage:
    python -m src.Run_Experiment_Node --config config.yaml --input 0 --output results/
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch_geometric.data import Data

from utils import load_config, set_seed, dataset_func, get_save_path, compute_fidelity_minus, compute_direct_constraint_coverage
from model import get_model
from apxchase import ApxChase
from exhaustchase import ExhaustChase
from constraints import get_constraints
from constraint_mining import resolve_constraints
from Edge_masking import mask_edges_for_node_classification
from baselines import run_gnn_explainer_node, PGExplainerBaseline

import time

try:
    from matcher import find_pattern_matches as _pattern_match_fn  # type: ignore
    from grounding_semantics import constraint_activation_summary as _constraint_activation_summary  # type: ignore
except Exception:
    _pattern_match_fn = None
    _constraint_activation_summary = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run node classification explanation pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--input", type=int, default=None, help="Target node index (or index in target_nodes list)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--run_all", action="store_true", help="Run on all target nodes")
    return parser.parse_args()


def _load_trained_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = get_model(config).to(device)
    model_path = os.path.join("models", f"{config['data_name']}_{config['model_name']}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def _node_witness_metrics(chaser: ApxChase, witness: Data, reference_graph: Data, device: torch.device) -> Dict[str, Any]:
    conc = float(chaser.conc_fn(witness))
    aln = float(getattr(witness, "_alignment", chaser.rpr_fn(witness)))
    fid_minus = float(compute_fidelity_minus(chaser.model, reference_graph, witness, device, is_node=True))
    q_score = float(chaser.alpha * conc + chaser.beta * aln)
    return {
        "num_nodes": int(witness.num_nodes),
        "num_edges": int(witness.edge_index.size(1)),
        "conc": conc,
        "alignment": aln,
        "q": q_score,
        "fidelity_minus": fid_minus,
        "delta_edges": list(getattr(witness, "delta_edges", [])),
        "supporting_edges": list(getattr(witness, "supporting_edges", [])),
        "grounded_constraints": list(getattr(witness, "grounded_constraints", [])),
        "grounding_details": list(getattr(witness, "_grounding_details", [])),
    }


def _constraint_signature(constraints: List[dict]) -> str:
    names = []
    for constraint in constraints:
        try:
            names.append(str(constraint.get("name", constraint)))
        except Exception:
            names.append(str(constraint))
    payload = json.dumps(sorted(names), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _observed_cache_dir(config: Dict[str, Any]) -> str:
    root = os.path.join("artifacts", "observed_graph_cache", str(config.get("data_name", "dataset")).lower())
    os.makedirs(root, exist_ok=True)
    return root


def _mask_ratio_token(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return str(value).replace(".", "p")
    except Exception:
        return str(value)


def _prepare_observed_node_workload(
    data: Data,
    target_node: int,
    constraints: List[dict],
    config: Dict[str, Any],
) -> Tuple[Data, List[Tuple[int, int]], torch.Tensor]:
    cache_root = _observed_cache_dir(config)
    cache_key = (
        f"node_{int(target_node)}"
        f"__L_{int(config.get('L', 2))}"
        f"__max_masks_{int(config.get('max_masks', 1))}"
        f"__mask_ratio_{_mask_ratio_token(config.get('mask_ratio', None))}"
        f"__seed_{int(config.get('random_seed', 0))}"
        f"__pc_{1 if bool(config.get('preserve_connectivity', True)) else 0}"
        f"__constraints_{_constraint_signature(constraints)}.pt"
    )
    cache_path = os.path.join(cache_root, cache_key)
    if os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        return payload["observed_subgraph"], payload["dropped_edges"], payload["node_subset"]

    observed_subgraph, dropped_edges, node_subset = mask_edges_for_node_classification(
        data,
        target_node,
        constraints,
        num_hops=config.get("L", 2),
        max_masks=config.get("max_masks", 1),
        mask_ratio=config.get("mask_ratio", None),
        seed=config.get("random_seed"),
    )
    torch.save(
        {
            "observed_subgraph": observed_subgraph.cpu(),
            "dropped_edges": dropped_edges,
            "node_subset": node_subset.cpu(),
        },
        cache_path,
    )
    return observed_subgraph, dropped_edges, node_subset


def _run_one_node_apxchase(
    target_node: int,
    observed_subgraph: Data,
    dropped_edges: List[Tuple[int, int]],
    constraints: List[dict],
    config: Dict[str, Any],
    device: torch.device,
    chaser: ApxChase
) -> Tuple[float, int, float, float, float]:
    """Run ApxChase on a single target node."""

    masked_subgraph = observed_subgraph.clone()
    observed_num_edges = int(masked_subgraph.edge_index.size(1))
    masked_subgraph = masked_subgraph.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = chaser.model(masked_subgraph.x, masked_subgraph.edge_index)
        probs = torch.softmax(logits, dim=-1)
        y_ref = logits.argmax(dim=-1)
    
    masked_subgraph.y_ref = y_ref.detach()
    true_label = int(masked_subgraph.y[masked_subgraph.target_node_subgraph_id].item()) if hasattr(masked_subgraph, 'y') else None
    pred_label = int(y_ref[masked_subgraph.target_node_subgraph_id].item())
    
    # Run ApxChase/HeuChase/ExhaustChase
    t0 = time.time()
    result = chaser.explain_node(masked_subgraph, masked_subgraph.target_node_subgraph_id)
    t1 = time.time()
    
    # Handle different return values (ExhaustChase returns 3 values, others return 2)
    if len(result) == 3:
        Sigma_star, witnesses, enforce_time = result
    else:
        Sigma_star, witnesses = result
    
    elapsed = t1 - t0
    run_stats = dict(getattr(chaser, "_last_run_stats", {}) or {})
    candidate_count = int(run_stats.get("num_candidates_generated", 0) or 0)
    verified_witness_count = int(run_stats.get("num_candidates_verified", len(witnesses)) or 0)
    selected_witness_count = int(len(witnesses))
    admitted_candidate_count = int(run_stats.get("num_candidates_admitted", 0) or 0)
    
    # Extract coverage
    coverage_names = sorted(set([c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in Sigma_star]))
    
    # Compute metrics
    witness_summaries = []
    fidelity_scores = []
    conciseness_scores = []
    for w_idx, witness in enumerate(witnesses):
        witness_metric = _node_witness_metrics(chaser, witness, masked_subgraph, device)
        fidelity_scores.append(witness_metric["fidelity_minus"])
        num_edges = int(witness.edge_index.size(1))
        conciseness = 1.0 - (num_edges / observed_num_edges) if observed_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        
        witness_summaries.append({
            "index": w_idx,
            **witness_metric,
            "conciseness": float(conciseness),
        })
    
    avg_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if conciseness_scores else 0.0
    total_constraints = len(constraints)
    activation = _constraint_activation_summary(masked_subgraph, constraints, masked_subgraph, _pattern_match_fn) if _constraint_activation_summary else {"hit_names": set(), "active_names": set()}
    hit_names = sorted(activation["hit_names"])
    active_names = sorted(activation["active_names"])
    coverage_ratio_global = len(coverage_names) / total_constraints if total_constraints > 0 else 0.0
    coverage_ratio_normalized = len(coverage_names) / len(active_names) if active_names else 0.0
    set_objective = float(sum(w["q"] for w in witness_summaries) + chaser.gamma * coverage_ratio_global)
    
    # Save results
    save_root = config.get("save_dir") or get_save_path(config["data_name"], config.get("exp_name", "experiment"))
    os.makedirs(save_root, exist_ok=True)
    
    metrics = {
        "target_node": int(target_node),
        "target_node_subgraph_id": int(masked_subgraph.target_node_subgraph_id),
        "true_label": true_label,
        "predicted_label": pred_label,
        "prediction_confidence": probs[masked_subgraph.target_node_subgraph_id].tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "candidate_count": candidate_count,
        "verified_witness_count": verified_witness_count,
        "selected_witness_count": selected_witness_count,
        "admitted_candidate_count": admitted_candidate_count,
        "num_witnesses": verified_witness_count,
        "coverage_size": len(coverage_names),
        "covered_constraints": coverage_names,
        "covered_constraint_count": len(coverage_names),
        "hit_constraint_count": len(hit_names),
        "hit_constraints": hit_names,
        "active_constraint_count": len(active_names),
        "active_constraints": active_names,
        "total_constraints": total_constraints,
        "coverage_ratio": float(coverage_ratio_normalized),
        "coverage_ratio_global": float(coverage_ratio_global),
        "coverage_ratio_normalized": float(coverage_ratio_normalized),
        "witnesses": witness_summaries,
        "avg_fidelity_minus": avg_fidelity,
        "avg_conciseness": avg_conciseness,
        "original_num_edges": observed_num_edges,
        "set_coverage_ratio": float(coverage_ratio_normalized),
        "set_coverage_ratio_global": float(coverage_ratio_global),
        "set_coverage_ratio_normalized": float(coverage_ratio_normalized),
        "set_objective_F": set_objective,
        "runtime_sec": float(elapsed),
    }
    metrics.update(run_stats)
    
    with open(os.path.join(save_root, f"metrics_node_{target_node}.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    clean_subgraph = getattr(masked_subgraph, "_query_graph", None)
    if clean_subgraph is not None:
        torch.save(clean_subgraph.cpu(), os.path.join(save_root, f"clean_subgraph_node_{target_node}.pt"))
    torch.save(masked_subgraph.cpu(), os.path.join(save_root, f"observed_subgraph_node_{target_node}.pt"))
    
    print(
        f"[Node {target_node}] witnesses={verified_witness_count}, "
        f"selected_witnesses={selected_witness_count}, "
        f"covered_constraints={len(coverage_names)}/{max(1, len(active_names))} active "
        f"fid={avg_fidelity:.4f}, conc={avg_conciseness:.4f}, time={elapsed:.4f}s"
    )

    return elapsed, verified_witness_count, avg_fidelity, avg_conciseness, coverage_ratio_normalized


def _run_one_node_baseline(
    target_node: int,
    observed_subgraph: Data,
    dropped_edges: List[Tuple[int, int]],
    constraints: List[dict],
    config: Dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    baseline_name: str,
) -> Tuple[float, int, float, float, float]:
    """Run GNNExplainer or PGExplainer on a single target node."""

    masked_subgraph = observed_subgraph.clone()
    target_id = int(masked_subgraph.target_node_subgraph_id)
    subgraph = masked_subgraph.to(device)
    
    # Run baseline explainer
    t0 = time.time()
    if baseline_name == "gnnexplainer":
        from baselines import run_gnn_explainer_node
        result = run_gnn_explainer_node(
            model=model,
            data=subgraph,
            target_node=target_id,
            epochs=config.get("gnn_epochs", 100),
            device=device,
        )
    elif baseline_name == "pgexplainer":
        from baselines import run_pgexplainer_node
        result = run_pgexplainer_node(
            model=model,
            data=subgraph,
            target_node=target_id,
            epochs=config.get("pg_epochs", 30),
            device=device,
            full_data=subgraph,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    t1 = time.time()
    elapsed = t1 - t0
    
    # Extract edge mask and compute metrics
    edge_mask = result.get("edge_mask")
    if edge_mask is None:
        print(f"[Warning] {baseline_name} returned no edge_mask for node {target_node}")
        return elapsed, 0, 0.0, 0.0, 0.0
    
    # Top-k edges as explanation
    baseline_edge_topk = int(config.get("baseline_edge_topk", config.get("K", config.get("k", 10))))
    k = min(baseline_edge_topk, edge_mask.size(0))
    _, topk_indices = torch.topk(edge_mask, k=k)
    
    # Build explanation subgraph
    expl_edge_index = subgraph.edge_index[:, topk_indices].to(device)
    expl_subgraph = Data(
        x=subgraph.x,
        edge_index=expl_edge_index,
        y=subgraph.y,
        num_nodes=subgraph.num_nodes,
    )
    if hasattr(subgraph, "y_type") and subgraph.y_type is not None:
        expl_subgraph.y_type = subgraph.y_type
    if hasattr(subgraph, "node_labels") and subgraph.node_labels is not None:
        expl_subgraph.node_labels = subgraph.node_labels
    
    # Compute Fidelity-
    fid_minus = compute_fidelity_minus(model, subgraph, expl_subgraph, device, is_node=True, target_node_id=target_id)
    
    # Compute Conciseness
    original_num_edges = int(subgraph.edge_index.size(1))
    explanation_num_edges = int(expl_subgraph.edge_index.size(1))
    conciseness = 1.0 - (explanation_num_edges / original_num_edges) if original_num_edges > 0 else 0.0
    
    # Baselines only output G_s. They do not construct G_g, so a constraint is
    # counted as covered only when G_s itself directly satisfies it.
    coverage_stats = compute_direct_constraint_coverage(
        expl_subgraph,
        constraints,
        workload_graph=expl_subgraph.cpu(),
        return_stats=True,
    )
    covered_constraint_names = coverage_stats["covered_constraint_names"]
    coverage = coverage_stats["coverage_ratio_normalized"]
    
    # Save results
    save_root = config.get("save_dir") or get_save_path(config["data_name"], config.get("exp_name", "experiment"))
    os.makedirs(save_root, exist_ok=True)
    
    metrics = {
        "baseline": baseline_name,
        "target_node": int(target_node),
        "target_node_subgraph_id": target_id,
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "candidate_count": 1,
        "verified_witness_count": 1,
        "selected_witness_count": 1,
        "admitted_candidate_count": 1,
        "num_witnesses": 1,
        "predicted_label": int(result["pred"]),
        "prediction_confidence": result["prob"].tolist() if hasattr(result["prob"], "tolist") else float(result["prob"]),
        "original_num_edges": original_num_edges,
        "explanation_num_edges": explanation_num_edges,
        "top_k": k,
        "fidelity_minus": float(fid_minus),
        "conciseness": float(conciseness),
        "coverage_ratio": float(coverage),
        "coverage_ratio_global": float(coverage_stats["coverage_ratio_global"]),
        "coverage_ratio_normalized": float(coverage_stats["coverage_ratio_normalized"]),
        "hit_constraint_count": int(coverage_stats["hit_constraint_count"]),
        "hit_constraints": coverage_stats["hit_constraint_names"],
        "active_constraint_count": int(coverage_stats["active_constraint_count"]),
        "active_constraints": coverage_stats["active_constraint_names"],
        "covered_constraint_count": int(coverage_stats["covered_constraint_count"]),
        "delta_edges": list(getattr(expl_subgraph, "delta_edges", [])),
        "supporting_edges": list(getattr(expl_subgraph, "supporting_edges", [])),
        "grounded_constraints": list(getattr(expl_subgraph, "grounded_constraints", [])),
        "covered_constraints": covered_constraint_names,
        "total_constraints": len(constraints),
        "runtime_sec": float(elapsed),
    }
    
    with open(os.path.join(save_root, f"metrics_node_{target_node}_{baseline_name}.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    torch.save({
        "edge_mask": edge_mask.cpu(),
        "explanation_edges": expl_edge_index.cpu(),
        "subgraph": subgraph.cpu(),
    }, os.path.join(save_root, f"expl_node_{target_node}_{baseline_name}.pt"))
    
    print(f"[{baseline_name.upper()} Node {target_node}] fid={fid_minus:.4f}, conc={conciseness:.4f}, time={elapsed:.4f}s")
    
    return elapsed, 1, float(fid_minus), float(conciseness), coverage


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    set_seed(config.get("random_seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = args.output or config.get("save_dir") or get_save_path(config["data_name"], config.get("exp_name", "experiment"))
    config["save_dir"] = save_root
    window_size = int(config.get("K", config.get("k", 10)))
    local_budget = int(config.get("local_budget_k", config.get("Budget", 4)))
    
    # Load dataset
    dataset_resource = dataset_func(config)
    if isinstance(dataset_resource, dict):
        data = dataset_resource['data']
        # Extract sampled target nodes from dataset_resource (BAShape, Yelp, etc.)
        sampled_targets = dataset_resource.get('target_nodes', [])
    else:
        data = dataset_resource
        sampled_targets = []
    
    # Load model and constraints
    model = _load_trained_model(config, device)
    constraints = resolve_constraints(config, dataset_resource, model=model, device=device, save_dir=save_root)
    
    # Initialize explainer based on exp_name
    exp_name = str(config.get("exp_name", "apxchase_cora")).lower()
    
    debug_mode = bool(config.get("debug", False))

    chaser = ApxChase(
        model=model,
        Sigma=constraints,
        L=config.get("L", 2),
        k=window_size,
        B=local_budget,
        alpha=config.get("alpha", 1.0),
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 1.0),
        seed_per_constraint=config.get("apx_seed_per_constraint", 2),
        candidate_expand_steps=config.get("apx_candidate_expand_steps", 2),
        candidate_branch_factor=config.get("apx_candidate_branch_factor", 3),
        candidate_beam_width=config.get("apx_candidate_beam_width", 6),
        candidate_max_masks=config.get("apx_candidate_max_masks", 48),
        legacy_prefix_checkpoints=config.get("apx_legacy_prefix_checkpoints", 6),
        use_ranked_candidate_prioritization=config.get("apx_use_ranked_candidate_prioritization", True),
        use_task_aware_hybrid_generation=config.get("apx_use_task_aware_hybrid_generation", False),
        ranking_pool_factor=config.get("apx_ranking_pool_factor", 3),
        ranking_diversity_bonus=config.get("apx_ranking_diversity_bonus", 0.2),
        max_near_full_candidates=config.get("apx_max_near_full_candidates", 16),
        near_full_delete_budget=config.get("apx_near_full_delete_budget", 3),
        near_full_branch_factor=config.get("apx_near_full_branch_factor", 6),
        near_full_beam_width=config.get("apx_near_full_beam_width", 4),
        debug=debug_mode,
    )
    
    # Determine target nodes: prioritize dataset-sampled targets over config
    target_nodes_config = sampled_targets if sampled_targets else config.get("target_nodes", [])
    if args.run_all:
        target_nodes = target_nodes_config
    else:
        node_idx = args.input if args.input is not None else config.get("graph_index", 0)
        if node_idx < len(target_nodes_config):
            target_nodes = [target_nodes_config[node_idx]]
        else:
            # Fallback: use test nodes
            test_indices = torch.where(data.test_mask)[0]
            if node_idx < len(test_indices):
                target_nodes = [int(test_indices[node_idx].item())]
            else:
                raise IndexError(f"Node index {node_idx} out of range")
    
    # Run experiments
    total_time = 0.0
    total_expl = 0
    fidelity_scores = []
    conciseness_scores = []
    coverage_scores = []
    
    print(f"\n{'='*70}")
    print(f"Running {exp_name} on {config['data_name']}")
    print(f"Target nodes: {target_nodes}")
    print(f"{'='*70}\n")
    
    if exp_name.startswith("apxchase"):
        for target_node in target_nodes:
            observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, target_node, constraints, config)
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, observed_subgraph, dropped_edges, constraints, config, device, chaser
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("heuchase"):
        from heuchase import HeuChase
        chaser = HeuChase(
            model=model,
            Sigma=constraints,
            L=config.get("L", 2),
            k=window_size,
            B=local_budget,
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            m=config.get("heuchase_m", 6),
            noise_std=config.get("heuchase_noise_std", 1e-3),
            seed_per_constraint=config.get("apx_seed_per_constraint", 2),
            candidate_expand_steps=config.get("apx_candidate_expand_steps", 2),
            candidate_branch_factor=config.get("apx_candidate_branch_factor", 3),
            candidate_beam_width=config.get("apx_candidate_beam_width", 6),
            candidate_max_masks=config.get("apx_candidate_max_masks", 48),
            legacy_prefix_checkpoints=config.get("apx_legacy_prefix_checkpoints", 6),
            use_ranked_candidate_prioritization=config.get("apx_use_ranked_candidate_prioritization", True),
            use_task_aware_hybrid_generation=config.get("apx_use_task_aware_hybrid_generation", False),
            ranking_pool_factor=config.get("apx_ranking_pool_factor", 3),
            ranking_diversity_bonus=config.get("apx_ranking_diversity_bonus", 0.2),
            max_near_full_candidates=config.get("apx_max_near_full_candidates", 16),
            near_full_delete_budget=config.get("apx_near_full_delete_budget", 3),
            near_full_branch_factor=config.get("apx_near_full_branch_factor", 6),
            near_full_beam_width=config.get("apx_near_full_beam_width", 4),
            debug=debug_mode,
        )
        for target_node in target_nodes:
            observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, target_node, constraints, config)
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, observed_subgraph, dropped_edges, constraints, config, device, chaser
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("exhaustchase"):
        chaser = ExhaustChase(
            model=model,
            Sigma=constraints,
            L=config.get("L", 2),
            k=window_size,
            B=local_budget,
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            seed_per_constraint=config.get("apx_seed_per_constraint", 2),
            candidate_expand_steps=config.get("apx_candidate_expand_steps", 2),
            candidate_branch_factor=config.get("apx_candidate_branch_factor", 3),
            candidate_beam_width=config.get("apx_candidate_beam_width", 6),
            candidate_max_masks=config.get("apx_candidate_max_masks", 48),
            legacy_prefix_checkpoints=config.get("apx_legacy_prefix_checkpoints", 6),
            use_ranked_candidate_prioritization=config.get("apx_use_ranked_candidate_prioritization", True),
            use_task_aware_hybrid_generation=config.get("apx_use_task_aware_hybrid_generation", False),
            ranking_pool_factor=config.get("apx_ranking_pool_factor", 3),
            ranking_diversity_bonus=config.get("apx_ranking_diversity_bonus", 0.2),
            max_near_full_candidates=config.get("apx_max_near_full_candidates", 16),
            near_full_delete_budget=config.get("apx_near_full_delete_budget", 3),
            near_full_branch_factor=config.get("apx_near_full_branch_factor", 6),
            near_full_beam_width=config.get("apx_near_full_beam_width", 4),
            max_enforce_iterations=config.get("max_enforce_iterations", 100),
            debug=debug_mode,
        )
        for target_node in target_nodes:
            observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, target_node, constraints, config)
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, observed_subgraph, dropped_edges, constraints, config, device, chaser
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("gnnexplainer"):
        from baselines import run_gnn_explainer_node
        for target_node in target_nodes:
            observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, target_node, constraints, config)
            elapsed, count, fid, conc, cov = _run_one_node_baseline(
                target_node, observed_subgraph, dropped_edges, constraints, config, device, model, "gnnexplainer"
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("pgexplainer"):
        from baselines import run_pgexplainer_node
        for target_node in target_nodes:
            observed_subgraph, dropped_edges, _ = _prepare_observed_node_workload(data, target_node, constraints, config)
            elapsed, count, fid, conc, cov = _run_one_node_baseline(
                target_node, observed_subgraph, dropped_edges, constraints, config, device, model, "pgexplainer"
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    else:
        print(f"[Warning] Baseline {exp_name} not yet implemented for node classification")
        print(f"Supported: apxchase, heuchase, exhaustchase, gnnexplainer, pgexplainer")
        return
    
    # Print aggregate statistics
    print(f"\n{'='*70}")
    print(f"Aggregate Statistics")
    print(f"{'='*70}")
    print(f"Nodes processed: {len(target_nodes)}")
    print(f"Total explanations: {total_expl}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg time per node: {total_time / len(target_nodes):.4f}s" if target_nodes else "N/A")
    
    if fidelity_scores:
        print(f"\nFidelity- : avg={np.mean(fidelity_scores):.4f}, std={np.std(fidelity_scores):.4f}")
    if conciseness_scores:
        print(f"Conciseness: avg={np.mean(conciseness_scores):.4f}, std={np.std(conciseness_scores):.4f}")
    if coverage_scores:
        print(f"Coverage   : avg={np.mean(coverage_scores):.4f} ({np.mean(coverage_scores)*100:.2f}%), std={np.std(coverage_scores):.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
