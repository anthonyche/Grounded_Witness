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

import matplotlib.pyplot as plt
import networkx as nx

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
from apxchase_mutag import ApxChase as ApxChaseMUTAG  # Import MUTAG-specific version
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


# ----------------------- Visualization Utilities ----------------------- #

def _edge_index_to_undirected_edge_set(edge_index: torch.Tensor) -> set:
    """Convert a (2, E) edge_index tensor into a set of undirected edges (u, v) with u<=v."""
    if edge_index is None:
        return set()
    ei = edge_index.detach().cpu().numpy()
    E = ei.shape[1]
    s = set()
    for j in range(E):
        u = int(ei[0, j])
        v = int(ei[1, j])
        if u == v:
            # keep self-loops as (u, u)
            s.add((u, v))
        else:
            a, b = (u, v) if u <= v else (v, u)
            s.add((a, b))
    return s

def _build_nx_from_pyg(masked_graph: Data) -> nx.Graph:
    """Build a NetworkX Graph from a PyG Data graph (treated as undirected)."""
    G = nx.Graph()
    num_nodes = int(masked_graph.num_nodes if masked_graph.num_nodes is not None else masked_graph.x.size(0))
    G.add_nodes_from(range(num_nodes))
    base_edges = _edge_index_to_undirected_edge_set(masked_graph.edge_index)
    G.add_edges_from(base_edges)
    return G

def _draw_overlay_case(masked_graph: Data,
                       witness_graph: Data,
                       pos: dict,
                       out_path: str,
                       title: str = "",
                       clean_graph: Data = None) -> None:
    """
    Draw the complete original graph structure with colored nodes by atom type.
    Masked edges are shown as dashed light blue lines, witness edges in bold red.
    `pos` is a precomputed layout to keep node positions consistent.
    """
    # Use clean graph as base if provided, otherwise use masked graph
    if clean_graph is not None:
        G = _build_nx_from_pyg(clean_graph)
        # Calculate which edges were masked
        clean_edges = _edge_index_to_undirected_edge_set(clean_graph.edge_index)
        masked_edges_set = _edge_index_to_undirected_edge_set(masked_graph.edge_index)
        masked_edges = clean_edges - masked_edges_set
        remaining_edges = masked_edges_set
    else:
        G = _build_nx_from_pyg(masked_graph)
        masked_edges = set()
        remaining_edges = set(G.edges())
    
    # MUTAG node label mapping: 'C': 0, 'N': 1, 'O': 2, 'F': 3, 'I': 4, 'Cl': 5, 'Br': 6
    # 使用更柔和、饱和度更低的颜色
    atom_names = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    atom_colors = ['#F5DEB3', '#87CEEB', '#FFB6C1', '#98FB98', '#DDA0DD', '#90EE90', '#6A0DAD']
    # C: 淡黄色(Wheat), N: 天蓝色(SkyBlue), O: 浅粉色(LightPink), 
    # F: 淡绿色(PaleGreen), I: 淡紫色(Plum), Cl: 浅绿色(LightGreen), Br: 深紫色(DarkPurple)
    
    # Extract node labels from the graph's feature matrix
    # MUTAG uses one-hot encoding for atom types
    node_labels = []
    if hasattr(masked_graph, 'x') and masked_graph.x is not None:
        x_np = masked_graph.x.detach().cpu().numpy()
        # Assuming first 7 features are one-hot encoded atom types
        for i in range(len(x_np)):
            # Find which of the first 7 features is 1
            atom_idx = np.argmax(x_np[i, :7])
            node_labels.append(atom_idx)
    else:
        # Fallback: all nodes are carbon
        node_labels = [0] * G.number_of_nodes()
    
    # Assign colors based on atom type
    node_colors = [atom_colors[label] for label in node_labels]
    
    # Adjusted figure size (legend removed, so can use smaller width)
    plt.figure(figsize=(8, 6))
    
    # Draw nodes with atom-specific colors (alpha=1.0 for solid/opaque nodes)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, 
                          linewidths=1.5, edgecolors='black', alpha=1.0)
    
    # Draw only remaining edges (not masked edges) as solid grey background
    # Only draw edges that are NOT in masked_edges set
    if clean_graph is not None:
        remaining_edges_list = list(remaining_edges)
        if len(remaining_edges_list) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=remaining_edges_list,
                                  width=1.0, edge_color="#000000", alpha=0.7)
    else:
        # If no clean graph provided, draw all edges
        nx.draw_networkx_edges(G, pos, width=1.0, edge_color="#000000", alpha=0.7)
    
    # Masked edges are completely hidden (not drawn at all)
    # if len(masked_edges) > 0:
    #     nx.draw_networkx_edges(G, pos, edgelist=list(masked_edges),
    #                           width=2.5, edge_color="#4A90E2", alpha=0.8,
    #                           style='dashed')

    # Overlay witness edges in bold red (on top of everything)
    # IMPORTANT: Witness uses relabeled node IDs (0-indexed). 
    # We must map them back to original graph IDs using _nodes_in_full
    W_edges_relabeled = _edge_index_to_undirected_edge_set(witness_graph.edge_index)
    
    # Map witness edges back to original node IDs if mapping exists
    if hasattr(witness_graph, '_nodes_in_full') and witness_graph._nodes_in_full is not None:
        nodes_in_full = witness_graph._nodes_in_full.cpu().numpy()
        W_edges_original = set()
        for u_rel, v_rel in W_edges_relabeled:
            u_orig = int(nodes_in_full[u_rel])
            v_orig = int(nodes_in_full[v_rel])
            # Ensure canonical form (u <= v)
            if u_orig <= v_orig:
                W_edges_original.add((u_orig, v_orig))
            else:
                W_edges_original.add((v_orig, u_orig))
        
        # Debug: Print mapping info (can be removed later)
        if False:  # Set to True to enable debug output
            print(f"[VIZ DEBUG] Witness has {len(W_edges_relabeled)} edges (relabeled IDs)")
            print(f"[VIZ DEBUG] Mapped to {len(W_edges_original)} edges (original IDs)")
            print(f"[VIZ DEBUG] Node mapping: {nodes_in_full}")
    else:
        # Fallback: assume witness IDs match original IDs (unlikely for subgraphs)
        W_edges_original = W_edges_relabeled
        if False:  # Debug
            print(f"[VIZ WARN] No _nodes_in_full mapping found, using relabeled IDs as-is")
    
    # Show witness edges that exist in the original graph structure
    base_edges = set(G.edges())
    overlay_edges = list(W_edges_original & base_edges)
    
    # Debug: Check edge matching
    if False:  # Set to True to enable debug output
        print(f"[VIZ DEBUG] Base graph has {len(base_edges)} edges")
        print(f"[VIZ DEBUG] Witness (original IDs) has {len(W_edges_original)} edges")
        print(f"[VIZ DEBUG] Overlay (intersection) has {len(overlay_edges)} edges")
        if len(overlay_edges) < len(W_edges_original):
            missing = W_edges_original - base_edges
            print(f"[VIZ WARN] {len(missing)} witness edges not found in base graph: {list(missing)[:5]}...")
        
        # Check connectivity of witness subgraph
        if len(overlay_edges) > 0:
            witness_nx = nx.Graph()
            witness_nx.add_edges_from(overlay_edges)
            num_components = nx.number_connected_components(witness_nx)
            largest_cc = max(nx.connected_components(witness_nx), key=len) if num_components > 0 else set()
            print(f"[VIZ DEBUG] Witness connectivity: {num_components} component(s), largest={len(largest_cc)} nodes, total_edges={len(overlay_edges)}")
            if num_components > 1:
                print(f"[VIZ WARN] Witness has {num_components} DISCONNECTED components!")
                for idx, comp in enumerate(nx.connected_components(witness_nx), 1):
                    print(f"  Component {idx}: {len(comp)} nodes - {sorted(list(comp))}")

    if len(overlay_edges) > 0:
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=overlay_edges,
                               width=3.5,
                               edge_color="red",
                               alpha=0.95)
    
    # Draw repair edges in orange (hypothetical edges needed to complete constraints)
    # Repair edges are stored in witness_graph._repair_edges as list of (u, v) tuples
    # These are in full-graph coordinate system
    repair_edges = getattr(witness_graph, '_repair_edges', [])
    print(f"[VIZ DEBUG] Total repair edges for this witness: {len(repair_edges)}")
    if len(repair_edges) > 0:
        print(f"[VIZ DEBUG] Repair edges list: {repair_edges}")
        # Filter repair edges to show both concrete and hypothetical repairs
        valid_repair_edges = []
        witness_node_set = set()
        for u, v in overlay_edges:
            witness_node_set.add(u)
            witness_node_set.add(v)
        print(f"[VIZ DEBUG] Witness node set: {witness_node_set}")
        
        for edge in repair_edges:
            if isinstance(edge, (tuple, list)) and len(edge) == 2:
                u, v = edge
                
                # Case 1: Concrete repair edge (u, v) where both are real nodes
                if u >= 0 and v >= 0:
                    # Only show if edge exists in base graph but not in witness
                    has_node_u = G.has_node(u)
                    has_node_v = G.has_node(v)
                    print(f"[VIZ DEBUG] Checking concrete edge ({u}, {v}): has_node_u={has_node_u}, has_node_v={has_node_v}")
                    
                    if has_node_u and has_node_v:
                        has_edge = G.has_edge(u, v) or G.has_edge(v, u)
                        in_witness = (u, v) in overlay_edges or (v, u) in overlay_edges
                        u_in_witness = u in witness_node_set
                        v_in_witness = v in witness_node_set
                        print(f"[VIZ DEBUG]   has_edge={has_edge}, in_witness={in_witness}, u_in_witness={u_in_witness}, v_in_witness={v_in_witness}")
                        print(f"[VIZ DEBUG]   Clean graph edges from {u}: {list(G.neighbors(u))}")
                        print(f"[VIZ DEBUG]   Clean graph edges from {v}: {list(G.neighbors(v))}")
                        
                        if has_edge and not in_witness:
                            # Edge exists in clean graph but missing from witness - INTERNAL repair!
                            if u_in_witness and v_in_witness:
                                valid_repair_edges.append((u, v))
                                print(f"[VIZ] Concrete repair edge: ({u}, {v}) [INTERNAL - closes benzene ring]")
                            else:
                                valid_repair_edges.append((u, v))
                                print(f"[VIZ] Concrete repair edge: ({u}, {v}) [EXTERNAL]")
                        elif not has_edge:
                            print(f"[VIZ DEBUG]   Edge ({u}, {v}) does NOT exist in clean graph - this is a hypothetical repair")
                
                # Case 2: Hypothetical repair (u, -1) or (-1, v)
                # Find edges in clean graph from u (or to v) that could close the pattern
                elif u >= 0 and v == -1:
                    # Strategy: Find neighbors of u in clean graph
                    # Prefer neighbors that ARE in witness (to close internal patterns like benzene rings)
                    print(f"[VIZ DEBUG] Processing hypothetical edge ({u}, -1)")
                    if G.has_node(u):
                        print(f"[VIZ DEBUG]   Node {u} exists, neighbors: {list(G.neighbors(u))}")
                        # First try: find neighbors in witness that would close a missing edge
                        internal_candidates = []
                        external_candidates = []
                        
                        for neighbor in G.neighbors(u):
                            edge_exists = (u, neighbor) in overlay_edges or (neighbor, u) in overlay_edges
                            print(f"[VIZ DEBUG]   Checking neighbor {neighbor}: edge_exists={edge_exists}, in_witness={neighbor in witness_node_set}")
                            if not edge_exists:  # Edge missing in witness
                                if neighbor in witness_node_set:
                                    internal_candidates.append(neighbor)
                                else:
                                    external_candidates.append(neighbor)
                        
                        # Prefer internal edges (closes benzene rings etc.)
                        if internal_candidates:
                            neighbor = internal_candidates[0]
                            valid_repair_edges.append((u, neighbor))
                            print(f"[VIZ] Hypothetical repair edge from ({u}, -1): ({u}, {neighbor}) [INTERNAL]")
                        elif external_candidates:
                            neighbor = external_candidates[0]
                            valid_repair_edges.append((u, neighbor))
                            print(f"[VIZ] Hypothetical repair edge from ({u}, -1): ({u}, {neighbor}) [EXTERNAL]")
                
                elif u == -1 and v >= 0:
                    # Strategy: Find neighbors of v in clean graph
                    # Prefer neighbors that ARE in witness (to close internal patterns)
                    if G.has_node(v):
                        # First try: find neighbors in witness that would close a missing edge
                        internal_candidates = []
                        external_candidates = []
                        
                        for neighbor in G.neighbors(v):
                            edge_exists = (neighbor, v) in overlay_edges or (v, neighbor) in overlay_edges
                            if not edge_exists:  # Edge missing in witness
                                if neighbor in witness_node_set:
                                    internal_candidates.append(neighbor)
                                else:
                                    external_candidates.append(neighbor)
                        
                        # Prefer internal edges (closes benzene rings etc.)
                        if internal_candidates:
                            neighbor = internal_candidates[0]
                            valid_repair_edges.append((neighbor, v))
                            print(f"[VIZ] Hypothetical repair edge from (-1, {v}): ({neighbor}, {v}) [INTERNAL]")
                        elif external_candidates:
                            neighbor = external_candidates[0]
                            valid_repair_edges.append((neighbor, v))
                            print(f"[VIZ] Hypothetical repair edge from (-1, {v}): ({neighbor}, {v}) [EXTERNAL]")
        
        if len(valid_repair_edges) > 0:
            print(f"[VIZ] Drawing {len(valid_repair_edges)} repair edges in orange")
            nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist=valid_repair_edges,
                                   width=3.5,
                                   edge_color="orange",
                                   alpha=0.95)

    # Legend disabled - use separate legend file (mutag_legend.png)
    # from matplotlib.patches import Patch
    # from matplotlib.lines import Line2D
    # 
    # # Atom type legend
    # legend_elements = [Patch(facecolor=atom_colors[i], edgecolor='black', 
    #                          label=atom_names[i]) for i in range(len(atom_names))]
    # 
    # # Add separator
    # legend_elements.append(Line2D([0], [0], color='none', label=''))
    # 
    # # Edge type legend
    # legend_elements.append(Line2D([0], [0], color='#BBBBBB', linewidth=1.0, 
    #                               label='Original edges'))
    # if len(masked_edges) > 0:
    #     legend_elements.append(Line2D([0], [0], color='#4A90E2', linewidth=2.5, 
    #                                   linestyle='dashed', alpha=0.8,
    #                                   label='Masked out'))
    # legend_elements.append(Line2D([0], [0], color='red', linewidth=3.5, 
    #                               label='Witness'))
    # 
    # # Place legend outside the plot area on the right side
    # # Much larger font sizes for better readability in small figures
    # plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0),
    #           framealpha=0.95, fontsize=18, title='Legend', title_fontsize=20, ncol=1, 
    #           borderaxespad=0, edgecolor='gray', fancybox=True, 
    #           handlelength=2.5, handleheight=1.5)

    # Title removed as requested - figure will have no title
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


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
        preserve_connectivity=config.get("preserve_connectivity", True),
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

    # --- Case Study Visualization: base masked graph + overlays for all explanations in W_k --- #
    try:
        save_root = get_save_path(config["data_name"], config.get("exp_name", "experiment")) if config.get("save_dir") is None else config.get("save_dir")
        os.makedirs(save_root, exist_ok=True)

        # Compute layout on the COMPLETE original graph for better molecular structure
        clean_graph_for_viz = getattr(masked_graph, '_clean', base_graph)
        nx_base = _build_nx_from_pyg(clean_graph_for_viz)
        
        # Debug: Check masked_graph connectivity
        num_comps_base = nx.number_connected_components(nx_base)
        if num_comps_base > 1:
            print(f"[WARN] Masked graph has {num_comps_base} disconnected components!")
            for idx, comp in enumerate(nx.connected_components(nx_base), 1):
                print(f"  Masked Component {idx}: {len(comp)} nodes")
        
        # Use Kamada-Kawai layout for better molecular structure visualization
        # Falls back to spring layout if KK fails
        try:
            pos = nx.kamada_kawai_layout(nx_base)
        except:
            # Fallback to spring layout with tighter spacing
            pos = nx.spring_layout(nx_base, seed=42, k=0.5, iterations=50)

        # Draw **all** witnesses in the window W_k
        total_w = len(witnesses)
        for i, w in enumerate(witnesses, start=1):
            out_path = os.path.join(save_root, f"case_graph_{dataset_idx}_expl_{i}.png")
            _draw_overlay_case(
                masked_graph,
                w,
                pos,
                out_path,
                title=None,
                clean_graph=clean_graph_for_viz
            )
    except Exception as viz_e:
        print(f"[WARN] Visualization failed: {viz_e}")

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

    # Save the overlay edge lists for **all** witnesses (for reproducibility)
    try:
        all_edges_payload = []
        for i, w in enumerate(witnesses, start=1):
            edges_i = _edge_index_to_undirected_edge_set(w.edge_index)
            all_edges_payload.append({
                "index": i,
                "edges": sorted([list(e) for e in edges_i])
            })
        with open(os.path.join(save_root, f"case_graph_{dataset_idx}_expl_edges_all.json"), "w", encoding="utf-8") as fp:
            json.dump(all_edges_payload, fp, indent=2)
    except Exception as e_save:
        print(f"[WARN] Failed to save overlay edges: {e_save}")

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
        preserve_connectivity=config.get("preserve_connectivity", True),
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
        
        # Visualize the explanation
        try:
            # Use the same layout strategy as ApxChase for consistency
            clean_graph_for_viz = base_graph.cpu()
            nx_base = _build_nx_from_pyg(clean_graph_for_viz)
            
            # Use Kamada-Kawai layout for better molecular structure visualization
            try:
                pos = nx.kamada_kawai_layout(nx_base)
            except:
                # Fallback to spring layout with tighter spacing
                pos = nx.spring_layout(nx_base, seed=42, k=0.5, iterations=50)
            
            # Set _nodes_in_full to map witness nodes back to original graph
            # For GNNExplainer, the subgraph contains ALL nodes from masked_graph
            # So nodes_in_full is just [0, 1, 2, ..., n-1]
            num_nodes = int(subgraph.num_nodes if subgraph.num_nodes is not None else subgraph.x.size(0))
            setattr(subgraph, '_nodes_in_full', torch.arange(num_nodes))
            
            # Set empty repair edges for GNNExplainer (it doesn't generate repairs)
            setattr(subgraph, '_repair_edges', [])
            
            # Draw the case study visualization
            out_path = os.path.join(save_root, f"case_graph_{dataset_idx}_gnnexplainer.png")
            _draw_overlay_case(
                masked_graph=masked_graph.cpu(),
                witness_graph=subgraph.cpu(),
                pos=pos,
                out_path=out_path,
                title="",  # No title as per design
                clean_graph=clean_graph_for_viz
            )
            print(f"[GNNExplainer] Saved visualization to {out_path}")
        except Exception as e:
            print(f"[GNNExplainer] Failed to save visualization: {e}")
            import traceback
            traceback.print_exc()

    return elapsed, 1, fidelity_minus, conciseness, coverage_ratio  # treat one explanation per graph


def _run_one_graph_pgexplainer(pos: int, dataset_resource: Dict[str, Any], dataset: Any, constraints: List[dict], config: Dict[str, Any], device: torch.device, model: torch.nn.Module, pg_state: Dict[str, Any]) -> Tuple[float, int, float]:
    graph, dataset_idx = _select_test_graph(dataset_resource, dataset, pos)
    base_graph = _prepare_graph_for_model(graph)

    masked_graph, dropped_edges = mask_edges_by_constraints(
        base_graph,
        constraints,
        max_masks=config.get("max_masks", 1),
        seed=config.get("random_seed"),
        preserve_connectivity=config.get("preserve_connectivity", True),
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

    # 添加可视化：为PGExplainer生成子图图像
    if edge_mask is not None and config.get("visualize", True):
        try:
            # 使用与GNNExplainer相同的可视化流程
            k = config.get("pgexplainer_topk", 10)
            edge_mask_flat = edge_mask.flatten()
            topk_indices = torch.topk(edge_mask_flat, min(k, len(edge_mask_flat))).indices
            
            # 构建包含top-k边的子图
            selected_edges = masked_graph.edge_index[:, topk_indices]
            witness_graph = Data(
                x=masked_graph.x.clone(),
                edge_index=selected_edges,
                batch=masked_graph.batch.clone() if hasattr(masked_graph, 'batch') else None
            )
            
            # 添加节点映射和repair edges（PGExplainer不生成repairs）
            num_nodes = int(masked_graph.x.size(0))
            witness_graph._nodes_in_full = torch.arange(num_nodes)
            witness_graph._repair_edges = []
            
            # 使用与GNNExplainer相同的布局策略
            clean_graph_for_viz = base_graph.cpu()
            nx_base = _build_nx_from_pyg(clean_graph_for_viz)
            
            # 使用Kamada-Kawai布局（与ApxChase一致）
            try:
                pos = nx.kamada_kawai_layout(nx_base)
            except:
                pos = nx.spring_layout(nx_base, seed=42, k=0.5, iterations=50)
            
            # 保存可视化
            out_path = os.path.join(save_root, f"case_graph_{dataset_idx}_pgexplainer.png")
            _draw_overlay_case(
                masked_graph=masked_graph.cpu(),
                witness_graph=witness_graph.cpu(),
                pos=pos,
                out_path=out_path,
                title="",  # No title as per design
                clean_graph=clean_graph_for_viz
            )
            print(f"[PGExplainer] Saved visualization to {out_path}")
        except Exception as e:
            print(f"[PGExplainer] Failed to save visualization: {e}")
            import traceback
            traceback.print_exc()

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
        preserve_connectivity=config.get("preserve_connectivity", True),
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

    # Detect task type based on data_name
    data_name = config.get("data_name", "MUTAG")
    is_node_classification = data_name in ["Cora", "CiteSeer", "PubMed", "Yelp", "BAHouse"]
    
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
    
    # Use MUTAG-specific ApxChase with BFS-based multi-center strategy for connected explanations
    chaser = ApxChaseMUTAG(
        model=model,
        Sigma=constraints,
        L=config.get("L", 2),
        k=config.get("k", 10),
        B=config.get("Budget", 4),
        alpha=config.get("alpha", 1.0),
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 1.0),
        debug=True,  # Enable debug to diagnose connectivity issues
        num_centers=10,  # Not used anymore - we process all nodes as centers
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
        from heuchase_mutag import HeuChaseMUTAG
        chaser = HeuChaseMUTAG(
            model=model,
            Sigma=constraints,
            L=config.get("L", 2),
            k=config.get("k", 10),
            B=config.get("Budget", 4),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 1.0),
            m=config.get("heuchase_m", 6),
            noise_std=config.get("heuchase_noise_std", 0.1),  # Configurable noise level
            debug=True,  # Enable debug for case study
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