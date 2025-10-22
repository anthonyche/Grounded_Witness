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
from Edge_masking import mask_edges_for_node_classification
from baselines import run_gnn_explainer_node, PGExplainerBaseline

import time

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


def _run_one_node_apxchase(
    target_node: int,
    data: Data,
    constraints: List[dict],
    config: Dict[str, Any],
    device: torch.device,
    chaser: ApxChase
) -> Tuple[float, int, float, float, float]:
    """Run ApxChase on a single target node."""
    
    # Extract L-hop subgraph and apply constraint-based masking
    masked_subgraph, dropped_edges, node_subset = mask_edges_for_node_classification(
        data,
        target_node,
        constraints,
        num_hops=config.get("L", 2),
        max_masks=config.get("max_masks", 1),
        mask_ratio=config.get("mask_ratio", None),  # Use ratio if specified
        seed=config.get("random_seed"),
    )
    
    # Store clean version for repair cost calculation
    clean_subgraph = masked_subgraph.clone()
    clean_subgraph.edge_index = masked_subgraph.edge_index.clone()
    masked_subgraph._clean = clean_subgraph
    
    # Move to device
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
    
    # Extract coverage
    coverage_names = sorted(set([c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in Sigma_star]))
    
    # Compute metrics
    witness_summaries = []
    fidelity_scores = []
    conciseness_scores = []
    original_num_edges = int(masked_subgraph.edge_index.size(1))
    
    for w_idx, witness in enumerate(witnesses):
        conc = chaser.conc_fn(witness)
        rpr = chaser.rpr_fn(witness)
        num_edges = int(witness.edge_index.size(1))
        
        fid_minus = compute_fidelity_minus(chaser.model, masked_subgraph, witness, device, is_node=True)
        fidelity_scores.append(fid_minus)
        
        conciseness = 1.0 - (num_edges / original_num_edges) if original_num_edges > 0 else 0.0
        conciseness_scores.append(conciseness)
        
        witness_summaries.append({
            "index": w_idx,
            "num_nodes": int(witness.num_nodes),
            "num_edges": num_edges,
            "conc": float(conc),
            "rpr": float(rpr),
            "fidelity_minus": float(fid_minus),
            "conciseness": float(conciseness),
        })
    
    avg_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if conciseness_scores else 0.0
    coverage_ratio = len(coverage_names) / len(constraints) if constraints else 0.0
    
    # Save results
    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment"))
    os.makedirs(save_root, exist_ok=True)
    
    metrics = {
        "target_node": int(target_node),
        "target_node_subgraph_id": int(masked_subgraph.target_node_subgraph_id),
        "true_label": true_label,
        "predicted_label": pred_label,
        "prediction_confidence": probs[masked_subgraph.target_node_subgraph_id].tolist(),
        "num_dropped_edges": len(dropped_edges),
        "dropped_edges": dropped_edges,
        "num_witnesses": len(witnesses),
        "coverage_size": len(coverage_names),
        "covered_constraints": coverage_names,
        "total_constraints": len(constraints),
        "coverage_ratio": float(coverage_ratio),
        "witnesses": witness_summaries,
        "avg_fidelity_minus": avg_fidelity,
        "avg_conciseness": avg_conciseness,
        "original_num_edges": original_num_edges,
    }
    
    with open(os.path.join(save_root, f"metrics_node_{target_node}.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    torch.save(masked_subgraph.cpu(), os.path.join(save_root, f"masked_subgraph_node_{target_node}.pt"))
    
    print(f"[Node {target_node}] witnesses={len(witnesses)}, coverage={len(coverage_names)}/{len(constraints)}, "
          f"fid={avg_fidelity:.4f}, conc={avg_conciseness:.4f}, time={elapsed:.4f}s")
    
    return elapsed, len(witnesses), avg_fidelity, avg_conciseness, coverage_ratio


def _run_one_node_baseline(
    target_node: int,
    data: Data,
    config: Dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    baseline_name: str,
) -> Tuple[float, int, float, float, float]:
    """Run GNNExplainer or PGExplainer on a single target node."""
    
    # Extract L-hop subgraph (without constraint masking for fair comparison)
    from torch_geometric.utils import k_hop_subgraph, subgraph as pyg_subgraph
    
    # Get L-hop neighborhood nodes
    node_subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        target_node,
        num_hops=config.get("L", 2),
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        relabel_nodes=True,  # Relabel nodes to [0, num_nodes_in_subgraph)
    )
    
    # Verify edge_index is within bounds
    max_node_id = len(node_subset) - 1
    if edge_index.numel() > 0:
        # Ensure all edge indices are within [0, len(node_subset))
        valid_edge_mask = (edge_index[0] <= max_node_id) & (edge_index[1] <= max_node_id)
        edge_index = edge_index[:, valid_edge_mask]
    
    subgraph = Data(
        x=data.x[node_subset],
        edge_index=edge_index,
        y=data.y[node_subset] if hasattr(data, 'y') and data.y is not None else None,
        num_nodes=len(node_subset),
    )
    target_id = int(mapping.item())
    subgraph = subgraph.to(device)
    
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
            full_data=data,  # Pass full graph for training
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
    k = min(config.get("k", 10), edge_mask.size(0))
    _, topk_indices = torch.topk(edge_mask, k=k)
    
    # Build explanation subgraph
    expl_edge_index = subgraph.edge_index[:, topk_indices].to(device)
    expl_subgraph = Data(
        x=subgraph.x,
        edge_index=expl_edge_index,
        y=subgraph.y,
        num_nodes=subgraph.num_nodes,
    )
    
    # Compute Fidelity-
    fid_minus = compute_fidelity_minus(model, subgraph, expl_subgraph, device, is_node=True, target_node_id=target_id)
    
    # Compute Conciseness
    original_num_edges = int(subgraph.edge_index.size(1))
    explanation_num_edges = int(expl_subgraph.edge_index.size(1))
    conciseness = 1.0 - (explanation_num_edges / original_num_edges) if original_num_edges > 0 else 0.0
    
    # Compute Coverage - check if explanation subgraph satisfies constraints
    # Note: Baselines don't use constraints to generate explanations,
    # but we still evaluate whether the generated explanation satisfies constraints
    from constraints import get_constraints
    constraints = get_constraints(config.get("data_name", "Cora"))
    covered_constraint_names, coverage_ratio = compute_constraint_coverage(
        expl_subgraph, 
        constraints, 
        Budget=config.get("Budget", 8)
    )
    coverage = coverage_ratio
    
    # Save results
    save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment"))
    os.makedirs(save_root, exist_ok=True)
    
    metrics = {
        "baseline": baseline_name,
        "target_node": int(target_node),
        "target_node_subgraph_id": target_id,
        "predicted_label": int(result["pred"]),
        "prediction_confidence": result["prob"].tolist() if hasattr(result["prob"], "tolist") else float(result["prob"]),
        "original_num_edges": original_num_edges,
        "explanation_num_edges": explanation_num_edges,
        "top_k": k,
        "fidelity_minus": float(fid_minus),
        "conciseness": float(conciseness),
        "coverage_ratio": coverage,
        "covered_constraints": covered_constraint_names,
        "total_constraints": len(constraints),
        "elapsed_time": elapsed,
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
    constraints = get_constraints(config.get("data_name", "Cora"))
    
    # Initialize explainer based on exp_name
    exp_name = str(config.get("exp_name", "apxchase_cora")).lower()
    
    chaser = ApxChase(
        model=model,
        Sigma=constraints,
        L=config.get("L", 2),
        k=config.get("k", 10),
        B=config.get("Budget", 4),
        alpha=config.get("alpha", 1.0),
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 1.0),
        debug=True,
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
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, data, constraints, config, device, chaser
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
            k=config.get("k", 10),
            B=config.get("Budget", 4),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            m=config.get("heuchase_m", 6),
            debug=True,
        )
        for target_node in target_nodes:
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, data, constraints, config, device, chaser
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
            k=config.get("k", 10),
            B=config.get("Budget", 4),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            max_enforce_iterations=config.get("max_enforce_iterations", 100),
            debug=True,
        )
        for target_node in target_nodes:
            elapsed, count, fid, conc, cov = _run_one_node_apxchase(
                target_node, data, constraints, config, device, chaser
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("gnnexplainer"):
        from baselines import run_gnn_explainer_node
        for target_node in target_nodes:
            elapsed, count, fid, conc, cov = _run_one_node_baseline(
                target_node, data, config, device, model, "gnnexplainer"
            )
            total_time += elapsed
            total_expl += count
            fidelity_scores.append(fid)
            conciseness_scores.append(conc)
            coverage_scores.append(cov)
    
    elif exp_name.startswith("pgexplainer"):
        from baselines import run_pgexplainer_node
        for target_node in target_nodes:
            elapsed, count, fid, conc, cov = _run_one_node_baseline(
                target_node, data, config, device, model, "pgexplainer"
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
