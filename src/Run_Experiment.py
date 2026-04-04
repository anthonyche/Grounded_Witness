"""
Executable script that wires together the MUTAG explanation workflow:

1. Load configuration and pre-trained GNN.
2. Fetch a MUTAG graph instance and apply constraint-driven edge masking.
3. Run a forward pass to obtain the reference prediction.
4. Invoke ApxChase.explain_graph to generate witness subgraphs.
5. Compute summary metrics, print them, and persist artefacts to disk.

Usage:
    python -m src.Run_Experiment --config config.yaml --input 0 --output results/
"""

from __future__ import annotations
from itertools import count

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

from Edge_masking import mask_edges_by_constraints
from baselines import run_gnn_explainer_graph, PGExplainerBaseline

import time

try:
    # Optional debug-only matcher hook (may not be available in all setups).
    from matcher import find_pattern_matches as _pattern_match_fn  # type: ignore
    from grounding_semantics import constraint_activation_summary as _constraint_activation_summary  # type: ignore
except Exception:
    _pattern_match_fn = None
    _constraint_activation_summary = None

# Restrict OpenMP / BLAS thread usage to avoid shared-memory initialisation failures in sandboxed environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_HEAP_SEQ = count()


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


def _prepare_observed_graph_workload(
    pos: int,
    dataset_resource: Dict[str, Any],
    dataset: Any,
    constraints: List[dict],
    config: Dict[str, Any],
) -> Tuple[Data, Data, List[Tuple[int, int]], int, int | None]:
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    true_label = int(graph.y.item()) if hasattr(graph, "y") and graph.y is not None else None
    cache_root = _observed_cache_dir(config)
    cache_key = (
        f"graph_{int(dataset_idx)}"
        f"__max_masks_{int(config.get('max_masks', 1))}"
        f"__mask_ratio_{_mask_ratio_token(config.get('mask_ratio', None))}"
        f"__seed_{int(config.get('random_seed', 0))}"
        f"__pc_{1 if bool(config.get('preserve_connectivity', True)) else 0}"
        f"__constraints_{_constraint_signature(constraints)}.pt"
    )
    cache_path = os.path.join(cache_root, cache_key)

    base_graph = _prepare_graph_for_model(graph)
    if os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        return base_graph, payload["observed_graph"], payload["dropped_edges"], dataset_idx, true_label

    observed_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        mask_ratio=config.get("mask_ratio", None),
        seed=config.get("random_seed"),
        preserve_connectivity=config.get("preserve_connectivity", True),
    )
    observed_graph._clean = base_graph.clone()
    if hasattr(graph, "y") and graph.y is not None:
        observed_graph.y = graph.y.clone()
    observed_graph.E_base = observed_graph.edge_index.size(1)
    torch.save(
        {
            "observed_graph": observed_graph.cpu(),
            "dropped_edges": dropped_edges,
        },
        cache_path,
    )
    return base_graph, observed_graph, dropped_edges, dataset_idx, true_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUTAG witness generation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--input",
        type=int,
        default=0,
        help="Index within the test split to explain.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run explanations over the entire test split instead of a single index.",
    )
    return parser.parse_args()


def _select_test_graph(loaders: Dict[str, Any], dataset: Any, index: int) -> Tuple[Data, int]:
    """Pick the `index`-th graph from the test split and return the Data object plus its dataset id."""
    if "test_loader" not in loaders:
        raise ValueError("dataset_func did not return a test_loader for MUTAG.")
    test_subset = loaders["test_loader"].dataset  # torch.utils.data.Subset
    if not hasattr(test_subset, "indices"):
        raise ValueError("Expected test_loader.dataset to be a Subset with .indices.")
    if index < 0 or index >= len(test_subset.indices):
        raise IndexError(f"graph-index {index} is out of range for the test split (size={len(test_subset.indices)}).")
    dataset_idx = int(test_subset.indices[index])
    graph = dataset[dataset_idx]
    if not isinstance(graph, Data):
        raise TypeError("Expected dataset elements to be torch_geometric.data.Data objects.")
    return graph, dataset_idx


def _prepare_graph_for_model(graph: Data) -> Data:
    """Ensure the Data object carries a batch vector and resides on CPU (before device transfer)."""
    graph = graph.clone()
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


def _load_trained_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = get_model(config).to(device)
    model_path = os.path.join("models", f"{config['data_name']}_{config['model_name']}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained checkpoint not found at {model_path}. "
                                "Please train the model before running explanations.")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint formats:
    # 1. Direct state_dict (old format)
    # 2. Dict with 'model_state_dict' key (new format from HPC training scripts)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def _graph_to_device(graph: Data, device: torch.device) -> Data:
    graph = graph.to(device)
    graph.batch = graph.batch.to(device)
    if hasattr(graph, "y"):
        graph.y = graph.y.to(device)
    return graph


def _debug_list_constraints(constraints: List[dict]) -> None:
    print(f"[DEBUG] Loaded {len(constraints)} constraints:")
    for i, c in enumerate(constraints):
        name = c.get("name", f"constraint_{i}")
        consequent_edges = len(c.get("consequent", {}).get("edges", [])) if isinstance(c.get("consequent"), dict) else "?"
        antecedent_edges = len(c.get("antecedent", {}).get("edges", [])) if isinstance(c.get("antecedent"), dict) else "?"
        print(f"  - {name} (consequent_edges={consequent_edges}, antecedent_edges={antecedent_edges})")

def _debug_scan_consequent_matches(graph: Data, constraints: List[dict], tag: str) -> None:
    if _pattern_match_fn is None:
        print(f"[DEBUG] Skipping consequent-match scan for '{tag}': matcher unavailable.")
        return
    try:
        num_nodes = int(graph.num_nodes)
        num_edges = int(graph.edge_index.size(1))
    except Exception:
        num_nodes = getattr(graph, "num_nodes", "?")
        num_edges = getattr(getattr(graph, "edge_index", None), "size", lambda *_: ["?","?"])(1)
    print(f"[DEBUG] Consequent-match scan on '{tag}' graph (|V|={num_nodes}, |E|={num_edges})")
    for c in constraints:
        name = c.get("name", "?")
        try:
            matches = _pattern_match_fn(graph, c.get("consequent", {}))
            print(f"    - {name}: consequent matches = {len(matches)}")
        except Exception as e:
            print(f"    - {name}: consequent scan error: {e}")


def _witness_metrics(chaser: ApxChase, witness: Data, reference_graph: Data, device: torch.device) -> Dict[str, Any]:
    conc = float(chaser.conc_fn(witness))
    aln = float(getattr(witness, "_alignment", chaser.rpr_fn(witness)))
    num_edges = int(witness.edge_index.size(1))
    fid_minus = float(compute_fidelity_minus(chaser.model, reference_graph, witness, device))
    q_score = float(chaser.alpha * conc + chaser.beta * aln)
    return {
        "num_nodes": int(witness.num_nodes if witness.num_nodes is not None else witness.x.size(0)),
        "num_edges": num_edges,
        "conc": conc,
        "alignment": aln,
        "q": q_score,
        "fidelity_minus": fid_minus,
        "delta_edges": list(getattr(witness, "delta_edges", [])),
        "supporting_edges": list(getattr(witness, "supporting_edges", [])),
        "grounded_constraints": list(getattr(witness, "grounded_constraints", [])),
        "grounding_details": list(getattr(witness, "_grounding_details", [])),
    }


# === Helper functions for running a single graph for each experiment type ===
def _run_one_graph_apxchase(base_graph: Data, observed_graph: Data, dropped_edges: List[Tuple[int, int]], dataset_idx: int, true_label: int | None, constraints: List[dict], config: Dict[str, Any], device: torch.device, chaser: ApxChase) -> Tuple[float, int]:
    _debug_scan_consequent_matches(base_graph, constraints, tag="original")
    masked_graph = observed_graph.clone()
    _debug_scan_consequent_matches(masked_graph, constraints, tag="masked")
    masked_graph = _graph_to_device(masked_graph, device)
    with torch.no_grad():
        logits = chaser.model(masked_graph)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        y_ref = logits.argmax(dim=-1)
    masked_graph.y_ref = y_ref.detach()

    print(f"[DEBUG] Model logits: {logits.detach().cpu().numpy().tolist()}")
    print(f"[DEBUG] Class probabilities: {probs.detach().cpu().numpy().tolist()}")

    t0 = time.time()
    Sigma_star, witnesses = chaser.explain_graph(masked_graph)
    t1 = time.time()
    elapsed = t1 - t0
    run_stats = dict(getattr(chaser, "_last_run_stats", {}) or {})
    candidate_count = int(run_stats.get("num_candidates_generated", 0) or 0)
    verified_witness_count = int(run_stats.get("num_candidates_verified", len(witnesses)) or 0)
    selected_witness_count = int(len(witnesses))
    admitted_candidate_count = int(run_stats.get("num_candidates_admitted", 0) or 0)

    coverage_names: List[str] = []
    for constraint in Sigma_star:
        if isinstance(constraint, dict) and "name" in constraint:
            coverage_names.append(constraint["name"])
        else:
            coverage_names.append(str(constraint))
    coverage_names = sorted(set(coverage_names))

    print(f"[DEBUG] ApxChase.explain_graph runtime: {elapsed:.4f}s")
    print("=== MUTAG Witness Generation Summary ===")
    print(f"Graph idx (dataset): {dataset_idx}")
    if true_label is not None:
        print(f"True label: {true_label}")
    print(f"Predicted label (y_ref): {int(y_ref.item())}")
    print(f"Dropped edges (undirected): {dropped_edges}")
    print(f"Verified witness count: {verified_witness_count}")
    print(f"Selected witness count (|W_k|): {selected_witness_count}")
    if selected_witness_count == 0:
        print("[DEBUG] No selected witnesses were admitted. Hints:")
        print("  - Check if any constraint consequents match on the observed graph (see consequent-match scan above).")
        print("  - Consider increasing Budget B, or adjusting masking to remove an aromatic/structural edge.")
        print("  - Ensure matcher uses consequent-to-antecedent backchase direction when triggering repairs.")
    print(f"Covered constraints ({len(coverage_names)}): {coverage_names}")

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    witness_summaries: List[Dict[str, Any]] = []
    fidelity_scores: List[float] = []
    conciseness_scores: List[float] = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    for w_idx, witness in enumerate(witnesses):
        witness_metric = _witness_metrics(chaser, witness, masked_graph, device)
        fidelity_scores.append(witness_metric["fidelity_minus"])
        num_edges = int(witness.edge_index.size(1))
        conciseness = 1.0 - (num_edges / original_num_edges) if original_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        summary = {"index": w_idx, **witness_metric, "conciseness": float(conciseness)}
        witness_summaries.append(summary)

    # 计算平均 Fidelity- 和 Conciseness
    avg_fidelity = float(np.mean(fidelity_scores)) if len(fidelity_scores) > 0 else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if len(conciseness_scores) > 0 else 0.0
    
    # 计算 Coverage ratio（评估主口径为 normalized；set objective 仍保持 global 口径）
    total_constraints = len(constraints)
    activation = _constraint_activation_summary(masked_graph, constraints, masked_graph, _pattern_match_fn) if _constraint_activation_summary else {"hit_names": set(), "active_names": set()}
    hit_names = sorted(activation["hit_names"])
    active_names = sorted(activation["active_names"])
    coverage_ratio_global = len(coverage_names) / total_constraints if total_constraints > 0 else 0.0
    coverage_ratio_normalized = len(coverage_names) / len(active_names) if active_names else 0.0
    set_objective = float(sum(w["q"] for w in witness_summaries) + chaser.gamma * coverage_ratio_global)

    metrics: Dict[str, Any] = {
        "graph_dataset_index": dataset_idx,
        "true_label": true_label,
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
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
        "original_num_edges": original_num_edges,
        "set_coverage_ratio": float(coverage_ratio_normalized),
        "set_coverage_ratio_global": float(coverage_ratio_global),
        "set_coverage_ratio_normalized": float(coverage_ratio_normalized),
        "set_objective_F": set_objective,
        "runtime_sec": float(elapsed),
    }
    metrics.update(run_stats)

    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    torch.save(base_graph.cpu(), os.path.join(save_root, f"clean_graph_{dataset_idx}.pt"))
    torch.save(masked_graph.cpu(), os.path.join(save_root, f"observed_graph_{dataset_idx}.pt"))
    return elapsed, verified_witness_count, avg_fidelity, avg_conciseness, coverage_ratio_normalized


def _run_one_graph_gnnexplainer(observed_graph: Data, dropped_edges: List[Tuple[int, int]], dataset_idx: int, constraints: List[dict], config: Dict[str, Any], device: torch.device, model: torch.nn.Module) -> Tuple[float, int, float]:
    masked_graph = observed_graph.clone()
    masked_graph = _graph_to_device(masked_graph, device)

    with torch.no_grad():
        logits = model(masked_graph)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        y_ref = logits.argmax(dim=-1)

    t0 = time.time()
    res = run_gnn_explainer_graph(model, masked_graph, epochs=config.get("gnnexplainer_epochs", 100))
    t1 = time.time()
    elapsed = t1 - t0

    # 计算 Fidelity-, Conciseness 和 Coverage (使用edge_mask生成的子图)
    edge_mask = res.get("edge_mask")
    fidelity_minus = 0.0
    conciseness = 0.0
    coverage_ratio = 0.0
    covered_constraints = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    if edge_mask is not None:
        # 选择top-k的边构建解释子图
        k = config.get("gnnexplainer_topk", 10)
        edge_mask_flat = edge_mask.flatten()
        topk_indices = torch.topk(edge_mask_flat, min(k, len(edge_mask_flat))).indices
        
        # 构建包含top-k边的子图
        selected_edges = masked_graph.edge_index[:, topk_indices]
        subgraph = Data(
            x=masked_graph.x.clone(),
            edge_index=selected_edges,
            batch=masked_graph.batch.clone() if hasattr(masked_graph, 'batch') else None
        )
        
        # 计算fidelity
        fidelity_minus = compute_fidelity_minus(model, masked_graph, subgraph, device)
        
        # 计算 Conciseness: 1 - (解释边数 / 原图边数)
        num_explanation_edges = int(selected_edges.size(1))
        conciseness = 1.0 - (num_explanation_edges / original_num_edges) if original_num_edges > 0 else 0.0
        
        # Baselines do not construct G_g. We only count constraints directly
        # satisfied by the witness G_s itself.
        subgraph_cpu = subgraph.cpu()
        coverage_stats = compute_direct_constraint_coverage(
            subgraph_cpu,
            constraints,
            workload_graph=subgraph_cpu,
            return_stats=True,
        )
        covered_constraints = coverage_stats["covered_constraint_names"]
        coverage_ratio = coverage_stats["coverage_ratio_normalized"]

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    metrics = {
        "graph_dataset_index": int(dataset_idx),
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "method": "GNNExplainer",
        "candidate_count": 1,
        "verified_witness_count": 1,
        "selected_witness_count": 1,
        "admitted_candidate_count": 1,
        "num_witnesses": 1,
        "edge_mask_topk": int(res.get("k", 0)),
        "avg_fidelity_minus": float(fidelity_minus),
        "avg_conciseness": float(conciseness),
        "coverage_size": len(covered_constraints),
        "covered_constraints": covered_constraints,
        "hit_constraint_count": int(coverage_stats["hit_constraint_count"]) if edge_mask is not None else 0,
        "hit_constraints": coverage_stats["hit_constraint_names"] if edge_mask is not None else [],
        "active_constraint_count": int(coverage_stats["active_constraint_count"]) if edge_mask is not None else 0,
        "active_constraints": coverage_stats["active_constraint_names"] if edge_mask is not None else [],
        "covered_constraint_count": int(coverage_stats["covered_constraint_count"]) if edge_mask is not None else 0,
        "total_constraints": len(constraints),
        "coverage_ratio": float(coverage_ratio),
        "coverage_ratio_global": float(coverage_stats["coverage_ratio_global"]) if edge_mask is not None else 0.0,
        "coverage_ratio_normalized": float(coverage_stats["coverage_ratio_normalized"]) if edge_mask is not None else 0.0,
        "original_num_edges": original_num_edges,
        "runtime_sec": float(elapsed),
    }
    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Persist the raw mask for future analysis
    if edge_mask is not None:
        torch.save(edge_mask.detach().cpu(), os.path.join(save_root, f"edge_mask_gnnexplainer_{dataset_idx}.pt"))

    return elapsed, 1, fidelity_minus, conciseness, coverage_ratio  # treat one explanation per graph


def _run_one_graph_pgexplainer(observed_graph: Data, dropped_edges: List[Tuple[int, int]], dataset_idx: int, constraints: List[dict], config: Dict[str, Any], device: torch.device, model: torch.nn.Module, pg_state: Dict[str, Any]) -> Tuple[float, int, float]:
    masked_graph = observed_graph.clone()
    masked_graph = _graph_to_device(masked_graph, device)

    # Lazy-create a PGExplainer and (optionally) quick-fit once per run
    t_total_start = time.time()
    
    if pg_state.get("explainer") is None:
        print(f"[Run_Experiment] 创建新的 PGExplainer, epochs={config.get('pgexplainer_epochs', 20)}, lr={config.get('pgexplainer_lr', 0.003)}")
        pg = PGExplainerBaseline(model, epochs=config.get("pgexplainer_epochs", 20), lr=config.get("pgexplainer_lr", 0.003))
        pg_state["explainer"] = pg
    else:
        print(f"[Run_Experiment] 使用现有 PGExplainer 实例")
        pg = pg_state["explainer"]

    if not pg_state.get("fitted", False):
        print(f"[Run_Experiment] 在第一个图上进行快速拟合")
        t_fit_start = time.time()
        # Quick warm-up on the first graph when no loader is available
        _ = pg.explain_graph(masked_graph, quick_fit=True)
        t_fit_end = time.time()
        pg_state["fitted"] = True
        print(f"[Run_Experiment] 快速拟合总用时: {t_fit_end - t_fit_start:.4f}秒")
    else:
        print(f"[Run_Experiment] PGExplainer 已经训练过，跳过拟合步骤")

    with torch.no_grad():
        logits = model(masked_graph)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        y_ref = logits.argmax(dim=-1)

    print(f"[Run_Experiment] 开始生成解释")
    t0 = time.time()
    res = pg.explain_graph(masked_graph)
    t1 = time.time()
    
    t_total_end = time.time()
    total_time = t_total_end - t_total_start
    explain_time = t1 - t0
    
    print(f"[Run_Experiment] 解释生成用时: {explain_time:.4f}秒, 总流程用时: {total_time:.4f}秒")
    
    elapsed = explain_time  # 保持与原代码一致，只记录解释时间

    # 计算 Fidelity-, Conciseness 和 Coverage (使用edge_mask生成的子图)
    edge_mask = res.get("edge_mask")
    fidelity_minus = 0.0
    conciseness = 0.0
    coverage_ratio = 0.0
    covered_constraints = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    if edge_mask is not None:
        # 选择top-k的边构建解释子图
        k = config.get("pgexplainer_topk", 10)
        edge_mask_flat = edge_mask.flatten()
        topk_indices = torch.topk(edge_mask_flat, min(k, len(edge_mask_flat))).indices
        
        # 构建包含top-k边的子图
        selected_edges = masked_graph.edge_index[:, topk_indices]
        subgraph = Data(
            x=masked_graph.x.clone(),
            edge_index=selected_edges,
            batch=masked_graph.batch.clone() if hasattr(masked_graph, 'batch') else None
        )
        
        # 计算fidelity
        fidelity_minus = compute_fidelity_minus(model, masked_graph, subgraph, device)
        
        # 计算 Conciseness: 1 - (解释边数 / 原图边数)
        num_explanation_edges = int(selected_edges.size(1))
        conciseness = 1.0 - (num_explanation_edges / original_num_edges) if original_num_edges > 0 else 0.0
        
        # Baselines do not construct G_g. We only count constraints directly
        # satisfied by the witness G_s itself.
        subgraph_cpu = subgraph.cpu()
        coverage_stats = compute_direct_constraint_coverage(
            subgraph_cpu,
            constraints,
            workload_graph=subgraph_cpu,
            return_stats=True,
        )
        covered_constraints = coverage_stats["covered_constraint_names"]
        coverage_ratio = coverage_stats["coverage_ratio_normalized"]

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    metrics = {
        "graph_dataset_index": int(dataset_idx),
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "method": "PGExplainer",
        "candidate_count": 1,
        "verified_witness_count": 1,
        "selected_witness_count": 1,
        "admitted_candidate_count": 1,
        "num_witnesses": 1,
        "edge_mask_topk": int(res.get("k", 0)),
        "avg_fidelity_minus": float(fidelity_minus),
        "avg_conciseness": float(conciseness),
        "coverage_size": len(covered_constraints),
        "covered_constraints": covered_constraints,
        "hit_constraint_count": int(coverage_stats["hit_constraint_count"]) if edge_mask is not None else 0,
        "hit_constraints": coverage_stats["hit_constraint_names"] if edge_mask is not None else [],
        "active_constraint_count": int(coverage_stats["active_constraint_count"]) if edge_mask is not None else 0,
        "active_constraints": coverage_stats["active_constraint_names"] if edge_mask is not None else [],
        "covered_constraint_count": int(coverage_stats["covered_constraint_count"]) if edge_mask is not None else 0,
        "total_constraints": len(constraints),
        "coverage_ratio": float(coverage_ratio),
        "coverage_ratio_global": float(coverage_stats["coverage_ratio_global"]) if edge_mask is not None else 0.0,
        "coverage_ratio_normalized": float(coverage_stats["coverage_ratio_normalized"]) if edge_mask is not None else 0.0,
        "original_num_edges": original_num_edges,
        "runtime_sec": float(elapsed),
    }
    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    edge_mask = res.get("edge_mask")
    if edge_mask is not None:
        torch.save(edge_mask.detach().cpu(), os.path.join(save_root, f"edge_mask_pgexplainer_{dataset_idx}.pt"))

    return elapsed, 1, fidelity_minus, conciseness, coverage_ratio


def _run_one_graph_exhaustchase(base_graph: Data, observed_graph: Data, dropped_edges: List[Tuple[int, int]], dataset_idx: int, true_label: int | None, constraints: List[dict], config: Dict[str, Any], device: torch.device, chaser: ExhaustChase, verbose: bool = False) -> Tuple[float, int]:
    """
    Run ExhaustChase on a single graph. Similar to _run_one_graph_apxchase,
    but includes the exhaustive enforcement overhead in the timing.
    """
    if verbose:
        _debug_scan_consequent_matches(base_graph, constraints, tag="original")
    masked_graph = observed_graph.clone()
    if verbose:
        _debug_scan_consequent_matches(masked_graph, constraints, tag="masked")
    masked_graph = _graph_to_device(masked_graph, device)
    with torch.no_grad():
        logits = chaser.model(masked_graph)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        y_ref = logits.argmax(dim=-1)
    masked_graph.y_ref = y_ref.detach()

    if verbose:
        print(f"[DEBUG] Model logits: {logits.detach().cpu().numpy().tolist()}")
        print(f"[DEBUG] Class probabilities: {probs.detach().cpu().numpy().tolist()}")

    # ExhaustChase returns enforce_time separately, but we include it in total time
    t0 = time.time()
    Sigma_star, witnesses, enforce_time = chaser.explain_graph(masked_graph)
    t1 = time.time()
    total_elapsed = t1 - t0
    candidate_gen_time = total_elapsed - enforce_time
    run_stats = dict(getattr(chaser, "_last_run_stats", {}) or {})
    candidate_count = int(run_stats.get("num_candidates_generated", 0) or 0)
    verified_witness_count = int(run_stats.get("num_candidates_verified", len(witnesses)) or 0)
    selected_witness_count = int(len(witnesses))
    admitted_candidate_count = int(run_stats.get("num_candidates_admitted", 0) or 0)

    coverage_names: List[str] = []
    for constraint in Sigma_star:
        if isinstance(constraint, dict) and "name" in constraint:
            coverage_names.append(constraint["name"])
        else:
            coverage_names.append(str(constraint))
    coverage_names = sorted(set(coverage_names))

    # Simplified output for batch processing
    print(
        f"[ExhaustChase] Graph {dataset_idx}: total={total_elapsed:.4f}s "
        f"(enforce={enforce_time:.4f}s, gen={candidate_gen_time:.4f}s), "
        f"witnesses={verified_witness_count}, selected_witnesses={selected_witness_count}, "
        f"coverage={len(coverage_names)}"
    )

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    witness_summaries: List[Dict[str, Any]] = []
    fidelity_scores: List[float] = []
    conciseness_scores: List[float] = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    for w_idx, witness in enumerate(witnesses):
        witness_metric = _witness_metrics(chaser, witness, masked_graph, device)
        fidelity_scores.append(witness_metric["fidelity_minus"])
        num_edges = int(witness.edge_index.size(1))
        conciseness = 1.0 - (num_edges / original_num_edges) if original_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        summary = {"index": w_idx, **witness_metric, "conciseness": float(conciseness)}
        witness_summaries.append(summary)

    # 计算平均 Fidelity- 和 Conciseness
    avg_fidelity = float(np.mean(fidelity_scores)) if len(fidelity_scores) > 0 else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if len(conciseness_scores) > 0 else 0.0
    
    # 计算 Coverage ratio
    total_constraints = len(constraints)
    activation = _constraint_activation_summary(masked_graph, constraints, masked_graph, _pattern_match_fn) if _constraint_activation_summary else {"hit_names": set(), "active_names": set()}
    hit_names = sorted(activation["hit_names"])
    active_names = sorted(activation["active_names"])
    coverage_ratio_global = len(coverage_names) / total_constraints if total_constraints > 0 else 0.0
    coverage_ratio_normalized = len(coverage_names) / len(active_names) if active_names else 0.0
    set_objective = float(sum(w["q"] for w in witness_summaries) + chaser.gamma * coverage_ratio_global)

    metrics: Dict[str, Any] = {
        "graph_dataset_index": dataset_idx,
        "true_label": true_label,
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
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
        "original_num_edges": original_num_edges,
        "enforce_time": float(enforce_time),
        "candidate_gen_time": float(candidate_gen_time),
        "total_time": float(total_elapsed),
        "set_coverage_ratio": float(coverage_ratio_normalized),
        "set_coverage_ratio_global": float(coverage_ratio_global),
        "set_coverage_ratio_normalized": float(coverage_ratio_normalized),
        "set_objective_F": set_objective,
        "runtime_sec": float(total_elapsed),
    }
    metrics.update(run_stats)

    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    torch.save(base_graph.cpu(), os.path.join(save_root, f"clean_graph_{dataset_idx}.pt"))
    torch.save(masked_graph.cpu(), os.path.join(save_root, f"observed_graph_{dataset_idx}.pt"))
    
    # Return total elapsed time (including enforcement overhead)
    return total_elapsed, verified_witness_count, avg_fidelity, avg_conciseness, coverage_ratio_normalized


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Detect task type based on data_name
    data_name = config.get("data_name", "MUTAG")
    is_node_classification = data_name in ["Cora", "CiteSeer", "PubMed", "Yelp", "BAHouse", "BAShape"]
    
    if is_node_classification:
        print(f"[Run_Experiment] Detected node classification task: {data_name}")
        print(f"[Run_Experiment] Redirecting to node classification pipeline...")
        from Run_Experiment_Node import main as node_main
        node_main()
        return

    # === Graph classification pipeline (MUTAG) ===
    print(f"[Run_Experiment] Detected graph classification task: {data_name}")
    
    graph_index = args.input if args.input is not None else config.get("graph_index", 0)
    max_masks = config.get("max_masks", 1)
    save_root = args.output or config.get("save_dir") or get_save_path(config["data_name"], config.get("exp_name", "default_experiment"))
    config["save_dir"] = save_root
    window_size = int(config.get("K", config.get("k", 10)))
    local_budget = int(config.get("local_budget_k", config.get("Budget", 4)))

    set_seed(config.get("random_seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load dataset assets (DataLoader dict for MUTAG).
    dataset_resource = dataset_func(config)
    if "test_loader" not in dataset_resource:
        raise RuntimeError("dataset_func is expected to return a dict with loaders for MUTAG.")
    test_subset = dataset_resource["test_loader"].dataset
    if not hasattr(test_subset, "indices"):
        raise ValueError("Expected test_loader.dataset to be a Subset with .indices.")

    dataset = dataset_resource["dataset"]

    # Shared resources: constraints, model, chaser
    model = _load_trained_model(config, device)
    constraints = resolve_constraints(config, dataset_resource, model=model, device=device, save_dir=save_root)
    _debug_list_constraints(constraints)
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
        debug=False,
    )

    exp_name = str(config.get("exp_name", "apxchase")).lower()

    # Decide which test positions to run
    test_subset = dataset_resource["test_loader"].dataset
    test_indices = list(test_subset.indices)
    configured_positions = config.get("graph_positions")
    if args.run_all:
        if configured_positions:
            test_positions = [int(pos) for pos in configured_positions]
        else:
            test_positions = list(range(len(test_indices)))
    else:
        graph_index = args.input if args.input is not None else config.get("graph_index", 0)
        if graph_index < 0 or graph_index >= len(test_indices):
            raise IndexError(f"graph-index {graph_index} is out of range for the test split (size={len(test_indices)}).")
        test_positions = [graph_index]

    total_time = 0.0
    total_expl = 0
    per_graph_counts: List[int] = []
    fidelity_scores: List[float] = []
    conciseness_scores: List[float] = []
    coverage_scores: List[float] = []

    if exp_name.startswith("apxchase"):  # original pipeline
        for pos in test_positions:
            base_graph, observed_graph, dropped_edges, dataset_idx, true_label = _prepare_observed_graph_workload(pos, dataset_resource, dataset, constraints, config)
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_apxchase(base_graph, observed_graph, dropped_edges, dataset_idx, true_label, constraints, config, device, chaser)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    elif exp_name.startswith("gnnexplainer"):  # GNNExplainer baseline on masked graphs
        for pos in test_positions:
            _, observed_graph, dropped_edges, dataset_idx, _ = _prepare_observed_graph_workload(pos, dataset_resource, dataset, constraints, config)
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_gnnexplainer(observed_graph, dropped_edges, dataset_idx, constraints, config, device, model)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    elif exp_name.startswith("pgexplainer"):  # PGExplainer baseline on masked graphs
        pg_state: Dict[str, Any] = {}
        for pos in test_positions:
            _, observed_graph, dropped_edges, dataset_idx, _ = _prepare_observed_graph_workload(pos, dataset_resource, dataset, constraints, config)
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_pgexplainer(observed_graph, dropped_edges, dataset_idx, constraints, config, device, model, pg_state)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    elif exp_name.startswith("exhaustchase"):  # ExhaustChase baseline with full enforcement
        exhaust_chaser = ExhaustChase(
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
            debug=False,
        )
        # Only show verbose output for single graph runs
        verbose = len(test_positions) == 1
        for pos in test_positions:
            base_graph, observed_graph, dropped_edges, dataset_idx, true_label = _prepare_observed_graph_workload(pos, dataset_resource, dataset, constraints, config)
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_exhaustchase(base_graph, observed_graph, dropped_edges, dataset_idx, true_label, constraints, config, device, exhaust_chaser, verbose=verbose)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
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
            debug=False,
        )
        for pos in test_positions:
            base_graph, observed_graph, dropped_edges, dataset_idx, true_label = _prepare_observed_graph_workload(pos, dataset_resource, dataset, constraints, config)
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_apxchase(base_graph, observed_graph, dropped_edges, dataset_idx, true_label, constraints, config, device, chaser)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    else:
        raise ValueError(f"Unknown exp_name '{exp_name}'. Expected one of: apxchase, exhaustchase, heuchase, gnnexplainer, pgexplainer")

    # === Final aggregate stats over the run ===
    num_graphs_run = len(test_positions)
    print("\n===== Aggregate Stats over Test Split Run =====")
    print(f"Graphs processed: {num_graphs_run}")
    print(f"Total explanations across graphs: {total_expl}")
    print(f"Total runtime (s): {total_time:.4f}")
    if num_graphs_run > 0:
        print(f"Avg time per graph (s): {total_time / num_graphs_run:.6f}")
    if total_expl > 0:
        print(f"Avg time per explanation (s): {total_time / total_expl:.6f}")
    print(f"Explanations per graph: {per_graph_counts}")
    
    # === Fidelity- Statistics ===
    if len(fidelity_scores) > 0:
        overall_avg_fidelity = float(np.mean(fidelity_scores))
        print(f"\n===== Fidelity- Statistics =====")
        print(f"Overall Average Fidelity-: {overall_avg_fidelity:.6f}")
        print(f"Fidelity- per graph: {[f'{f:.4f}' for f in fidelity_scores]}")
        print(f"Min Fidelity-: {min(fidelity_scores):.6f}")
        print(f"Max Fidelity-: {max(fidelity_scores):.6f}")
        print(f"Std Fidelity-: {float(np.std(fidelity_scores)):.6f}")
    
    # === Conciseness Statistics ===
    if len(conciseness_scores) > 0:
        overall_avg_conciseness = float(np.mean(conciseness_scores))
        print(f"\n===== Conciseness Statistics =====")
        print(f"Overall Average Conciseness: {overall_avg_conciseness:.6f}")
        print(f"Conciseness per graph: {[f'{c:.4f}' for c in conciseness_scores]}")
        print(f"Min Conciseness: {min(conciseness_scores):.6f}")
        print(f"Max Conciseness: {max(conciseness_scores):.6f}")
        print(f"Std Conciseness: {float(np.std(conciseness_scores)):.6f}")
    
    # === Coverage Statistics ===
    if len(coverage_scores) > 0:
        overall_avg_coverage = float(np.mean(coverage_scores))
        print(f"\n===== Coverage Statistics =====")
        print(f"Overall Average Coverage: {overall_avg_coverage:.6f} ({overall_avg_coverage*100:.2f}%)")
        print(f"Coverage per graph: {[f'{c:.4f}' for c in coverage_scores]}")
        print(f"Min Coverage: {min(coverage_scores):.6f} ({min(coverage_scores)*100:.2f}%)")
        print(f"Max Coverage: {max(coverage_scores):.6f} ({max(coverage_scores)*100:.2f}%)")
        print(f"Std Coverage: {float(np.std(coverage_scores)):.6f}")


if __name__ == "__main__":
    main()
