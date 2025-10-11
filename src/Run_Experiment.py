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
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch_geometric.data import Data

from utils import load_config, set_seed, dataset_func, get_save_path, compute_fidelity_minus, compute_constraint_coverage
from model import get_model
from apxchase import ApxChase
from exhaustchase import ExhaustChase
from constraints import get_constraints

from Edge_masking import mask_edges_by_constraints
from baselines import run_gnn_explainer_graph, PGExplainerBaseline

import time

try:
    # Optional debug-only matcher hook (may not be available in all setups).
    from matcher import find_head_matches as _head_match_fn  # type: ignore
except Exception:
    _head_match_fn = None

# Restrict OpenMP / BLAS thread usage to avoid shared-memory initialisation failures in sandboxed environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_HEAP_SEQ = count()


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
        head_edges = len(c.get("head", {}).get("edges", [])) if isinstance(c.get("head"), dict) else "?"
        body_edges = len(c.get("body", {}).get("edges", [])) if isinstance(c.get("body"), dict) else "?"
        print(f"  - {name} (head_edges={head_edges}, body_edges={body_edges})")

def _debug_scan_head_matches(graph: Data, constraints: List[dict], tag: str) -> None:
    if _head_match_fn is None:
        print(f"[DEBUG] Skipping head-match scan for '{tag}': matcher.find_matches not available.")
        return
    try:
        num_nodes = int(graph.num_nodes)
        num_edges = int(graph.edge_index.size(1))
    except Exception:
        num_nodes = getattr(graph, "num_nodes", "?")
        num_edges = getattr(getattr(graph, "edge_index", None), "size", lambda *_: ["?","?"])(1)
    print(f"[DEBUG] Head-match scan on '{tag}' graph (|V|={num_nodes}, |E|={num_edges})")
    for c in constraints:
        name = c.get("name", "?")
        head = c.get("head", {})
        try:
            # matcher.find_head_matches expects the full constraint (with a "head" key),
            # not just the head-pattern. Pass the entire constraint to avoid KeyError('head').
            matches = _head_match_fn(graph, c)  # returns: list of dict assignments
            print(f"    - {name}: head matches = {len(matches)}")
        except Exception as e:
            print(f"    - {name}: head scan error: {e}")


# === Helper functions for running a single graph for each experiment type ===
def _run_one_graph_apxchase(pos: int, dataset_resource: Dict[str, Any], dataset: Any, constraints: List[dict], config: Dict[str, Any], device: torch.device, chaser: ApxChase) -> Tuple[float, int]:
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    true_label = int(graph.y.item()) if hasattr(graph, "y") and graph.y is not None else None

    base_graph = _prepare_graph_for_model(graph)
    _debug_scan_head_matches(base_graph, constraints, tag="original")

    masked_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        seed=config.get("random_seed"),
    )
    _debug_scan_head_matches(masked_graph, constraints, tag="masked")
    masked_graph._clean = base_graph
    masked_graph.y = graph.y.clone()

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
    print(f"Witness count (|W_k|): {len(witnesses)}")
    if len(witnesses) == 0:
        print("[DEBUG] No witnesses were admitted. Hints:")
        print("  - Check if any constraint heads match on the masked graph (see head-match scan above).")
        print("  - Consider increasing Budget B, or adjusting masking to remove an aromatic/structural edge.")
        print("  - Ensure matcher uses HEAD->BODY (backchase) direction when triggering repairs.")
    print(f"Covered constraints ({len(coverage_names)}): {coverage_names}")

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    witness_summaries: List[Dict[str, Any]] = []
    fidelity_scores: List[float] = []
    conciseness_scores: List[float] = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    for w_idx, witness in enumerate(witnesses):
        conc = chaser.conc_fn(witness)
        rpr = chaser.rpr_fn(witness)
        num_edges = int(witness.edge_index.size(1))
        
        # 计算 Fidelity-
        fid_minus = compute_fidelity_minus(chaser.model, masked_graph, witness, device)
        fidelity_scores.append(fid_minus)
        
        # 计算 Conciseness: 1 - (witness边数 / 原图边数)
        conciseness = 1.0 - (num_edges / original_num_edges) if original_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        
        summary = {
            "index": w_idx,
            "num_nodes": int(witness.num_nodes if witness.num_nodes is not None else witness.x.size(0)),
            "num_edges": num_edges,
            "conc": float(conc),
            "rpr": float(rpr),
            "fidelity_minus": float(fid_minus),
            "conciseness": float(conciseness),
        }
        witness_summaries.append(summary)

    # 计算平均 Fidelity- 和 Conciseness
    avg_fidelity = float(np.mean(fidelity_scores)) if len(fidelity_scores) > 0 else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if len(conciseness_scores) > 0 else 0.0
    
    # 计算 Coverage ratio
    total_constraints = len(constraints)
    coverage_ratio = len(coverage_names) / total_constraints if total_constraints > 0 else 0.0

    metrics: Dict[str, Any] = {
        "graph_dataset_index": dataset_idx,
        "true_label": true_label,
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "num_witnesses": len(witnesses),
        "coverage_size": len(coverage_names),
        "covered_constraints": coverage_names,
        "total_constraints": total_constraints,
        "coverage_ratio": float(coverage_ratio),
        "witnesses": witness_summaries,
        "avg_fidelity_minus": avg_fidelity,
        "avg_conciseness": avg_conciseness,
        "original_num_edges": original_num_edges,
    }

    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    torch.save(masked_graph.cpu(), os.path.join(save_root, f"masked_graph_{dataset_idx}.pt"))
    return elapsed, len(witnesses), avg_fidelity, avg_conciseness, coverage_ratio


def _run_one_graph_gnnexplainer(pos: int, dataset_resource: Dict[str, Any], dataset: Any, constraints: List[dict], config: Dict[str, Any], device: torch.device, model: torch.nn.Module) -> Tuple[float, int, float]:
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    base_graph = _prepare_graph_for_model(graph)

    # Baselines also operate on masked graph for fairness
    masked_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        seed=config.get("random_seed"),
    )
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
        
        # 计算 Coverage: 使用与ApxChase相同的constraint matching逻辑
        Budget = config.get("Budget", 8)
        subgraph_cpu = subgraph.cpu()
        covered_constraints, coverage_ratio = compute_constraint_coverage(subgraph_cpu, constraints, Budget)

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    metrics = {
        "graph_dataset_index": int(dataset_idx),
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "method": "GNNExplainer",
        "edge_mask_topk": int(res.get("k", 0)),
        "avg_fidelity_minus": float(fidelity_minus),
        "avg_conciseness": float(conciseness),
        "coverage_size": len(covered_constraints),
        "covered_constraints": covered_constraints,
        "total_constraints": len(constraints),
        "coverage_ratio": float(coverage_ratio),
        "original_num_edges": original_num_edges,
    }
    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Persist the raw mask for future analysis
    if edge_mask is not None:
        torch.save(edge_mask.detach().cpu(), os.path.join(save_root, f"edge_mask_gnnexplainer_{dataset_idx}.pt"))

    return elapsed, 1, fidelity_minus, conciseness, coverage_ratio  # treat one explanation per graph


def _run_one_graph_pgexplainer(pos: int, dataset_resource: Dict[str, Any], dataset: Any, constraints: List[dict], config: Dict[str, Any], device: torch.device, model: torch.nn.Module, pg_state: Dict[str, Any]) -> Tuple[float, int, float]:
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    base_graph = _prepare_graph_for_model(graph)

    masked_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        seed=config.get("random_seed"),
    )
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
        
        # 计算 Coverage: 使用与ApxChase相同的constraint matching逻辑
        Budget = config.get("Budget", 8)
        subgraph_cpu = subgraph.cpu()
        covered_constraints, coverage_ratio = compute_constraint_coverage(subgraph_cpu, constraints, Budget)

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    metrics = {
        "graph_dataset_index": int(dataset_idx),
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "method": "PGExplainer",
        "edge_mask_topk": int(res.get("k", 0)),
        "avg_fidelity_minus": float(fidelity_minus),
        "avg_conciseness": float(conciseness),
        "coverage_size": len(covered_constraints),
        "covered_constraints": covered_constraints,
        "total_constraints": len(constraints),
        "coverage_ratio": float(coverage_ratio),
        "original_num_edges": original_num_edges,
    }
    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    edge_mask = res.get("edge_mask")
    if edge_mask is not None:
        torch.save(edge_mask.detach().cpu(), os.path.join(save_root, f"edge_mask_pgexplainer_{dataset_idx}.pt"))

    return elapsed, 1, fidelity_minus, conciseness, coverage_ratio


def _run_one_graph_exhaustchase(pos: int, dataset_resource: Dict[str, Any], dataset: Any, constraints: List[dict], config: Dict[str, Any], device: torch.device, chaser: ExhaustChase, verbose: bool = False) -> Tuple[float, int]:
    """
    Run ExhaustChase on a single graph. Similar to _run_one_graph_apxchase,
    but includes the exhaustive enforcement overhead in the timing.
    """
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    true_label = int(graph.y.item()) if hasattr(graph, "y") and graph.y is not None else None

    base_graph = _prepare_graph_for_model(graph)
    if verbose:
        _debug_scan_head_matches(base_graph, constraints, tag="original")

    masked_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        seed=config.get("random_seed"),
    )
    if verbose:
        _debug_scan_head_matches(masked_graph, constraints, tag="masked")
    masked_graph._clean = base_graph
    masked_graph.y = graph.y.clone()

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

    coverage_names: List[str] = []
    for constraint in Sigma_star:
        if isinstance(constraint, dict) and "name" in constraint:
            coverage_names.append(constraint["name"])
        else:
            coverage_names.append(str(constraint))
    coverage_names = sorted(set(coverage_names))

    # Simplified output for batch processing
    print(f"[ExhaustChase] Graph {dataset_idx}: total={total_elapsed:.4f}s (enforce={enforce_time:.4f}s, gen={candidate_gen_time:.4f}s), witnesses={len(witnesses)}, coverage={len(coverage_names)}")

    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
    os.makedirs(save_root, exist_ok=True)

    witness_summaries: List[Dict[str, Any]] = []
    fidelity_scores: List[float] = []
    conciseness_scores: List[float] = []
    original_num_edges = int(masked_graph.edge_index.size(1))
    
    for w_idx, witness in enumerate(witnesses):
        conc = chaser.conc_fn(witness)
        rpr = chaser.rpr_fn(witness)
        num_edges = int(witness.edge_index.size(1))
        
        # 计算 Fidelity-
        fid_minus = compute_fidelity_minus(chaser.model, masked_graph, witness, device)
        fidelity_scores.append(fid_minus)
        
        # 计算 Conciseness: 1 - (witness边数 / 原图边数)
        conciseness = 1.0 - (num_edges / original_num_edges) if original_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        
        summary = {
            "index": w_idx,
            "num_nodes": int(witness.num_nodes if witness.num_nodes is not None else witness.x.size(0)),
            "num_edges": num_edges,
            "conc": float(conc),
            "rpr": float(rpr),
            "fidelity_minus": float(fid_minus),
            "conciseness": float(conciseness),
        }
        witness_summaries.append(summary)

    # 计算平均 Fidelity- 和 Conciseness
    avg_fidelity = float(np.mean(fidelity_scores)) if len(fidelity_scores) > 0 else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if len(conciseness_scores) > 0 else 0.0
    
    # 计算 Coverage ratio
    total_constraints = len(constraints)
    coverage_ratio = len(coverage_names) / total_constraints if total_constraints > 0 else 0.0

    metrics: Dict[str, Any] = {
        "graph_dataset_index": dataset_idx,
        "true_label": true_label,
        "predicted_label": int(y_ref.item()),
        "prediction_confidence": probs.tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "num_witnesses": len(witnesses),
        "coverage_size": len(coverage_names),
        "covered_constraints": coverage_names,
        "total_constraints": total_constraints,
        "coverage_ratio": float(coverage_ratio),
        "witnesses": witness_summaries,
        "avg_fidelity_minus": avg_fidelity,
        "avg_conciseness": avg_conciseness,
        "original_num_edges": original_num_edges,
        "enforce_time": float(enforce_time),
        "candidate_gen_time": float(candidate_gen_time),
        "total_time": float(total_elapsed),
    }

    with open(os.path.join(save_root, f"metrics_graph_{dataset_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    torch.save(masked_graph.cpu(), os.path.join(save_root, f"masked_graph_{dataset_idx}.pt"))
    
    # Return total elapsed time (including enforcement overhead)
    return total_elapsed, len(witnesses), avg_fidelity, avg_conciseness, coverage_ratio


def main() -> None:
    args = parse_args()
    config = load_config(args.config)


    graph_index = args.input if args.input is not None else config.get("graph_index", 0)
    max_masks = config.get("max_masks", 1)
    save_root = args.output or config.get("save_dir") or get_save_path(config["data_name"], config.get("exp_name", "default_experiment"))

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
    constraints = get_constraints(config.get("data_name", "MUTAG"))
    _debug_list_constraints(constraints)

    model = _load_trained_model(config, device)
    chaser = ApxChase(
        model=model,
        Sigma=constraints,
        L=config.get("L", 2),
        k=config.get("k", 10),
        B=config.get("Budget", 4),
        alpha=config.get("alpha", 1.0),
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 1.0),
        debug=False,
    )

    exp_name = str(config.get("exp_name", "apxchase_mutag")).lower()

    # Decide which test positions to run
    test_subset = dataset_resource["test_loader"].dataset
    test_indices = list(test_subset.indices)
    if args.run_all:
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
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_apxchase(pos, dataset_resource, dataset, constraints, config, device, chaser)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    elif exp_name.startswith("gnnexplainer"):  # GNNExplainer baseline on masked graphs
        for pos in test_positions:
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_gnnexplainer(pos, dataset_resource, dataset, constraints, config, device, model)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    elif exp_name.startswith("pgexplainer"):  # PGExplainer baseline on masked graphs
        pg_state: Dict[str, Any] = {}
        for pos in test_positions:
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_pgexplainer(pos, dataset_resource, dataset, constraints, config, device, model, pg_state)
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
            k=config.get("k", 10),
            B=config.get("Budget", 4),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            max_enforce_iterations=config.get("max_enforce_iterations", 100),
            debug=False,
        )
        # Only show verbose output for single graph runs
        verbose = len(test_positions) == 1
        for pos in test_positions:
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_exhaustchase(pos, dataset_resource, dataset, constraints, config, device, exhaust_chaser, verbose=verbose)
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
            k=config.get("k", 10),
            B=config.get("Budget", 4),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            m=config.get("heuchase_m", 6),
            debug=False,
        )
        for pos in test_positions:
            elapsed, count, avg_fid, avg_conc, cov = _run_one_graph_apxchase(pos, dataset_resource, dataset, constraints, config, device, chaser)
            total_time += elapsed
            total_expl += count
            per_graph_counts.append(count)
            fidelity_scores.append(avg_fid)
            conciseness_scores.append(avg_conc)
            coverage_scores.append(cov)

    else:
        raise ValueError(f"Unknown exp_name '{exp_name}'. Expected one of: apxchase_mutag, exhaustchase_mutag, heuchase_mutag, gnnexplainer_mutag, pgexplainer_mutag")

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
