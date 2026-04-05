from __future__ import annotations

"""
Constraint-driven edge masking used to build a single observed graph per workload.

Standard backchase semantics:
  - each constraint is φ = (P, c)
  - masking must preserve consequent matches for c
  - masking preferentially removes already bound edges from P (antecedent),
    so backchase has to recover P from the same node set
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import random

import torch
from torch_geometric.data import Data

try:
    from src.matcher import find_pattern_matches  # type: ignore
except Exception:
    from matcher import find_pattern_matches  # type: ignore

try:
    from src.connectivity_utils import select_edges_preserving_connectivity  # type: ignore
except Exception:
    from connectivity_utils import select_edges_preserving_connectivity  # type: ignore

try:
    from src.constraints import TGD  # type: ignore
except Exception:
    TGD = Dict[str, Any]


def _as_undirected(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _parse_edge_spec(edge: Any) -> Tuple[Any, Any, Any]:
    if not isinstance(edge, (tuple, list)):
        raise ValueError(f"Unsupported edge spec: {edge}")
    if len(edge) == 2:
        return edge[0], edge[1], None
    if len(edge) == 3:
        return edge[0], edge[1], edge[2]
    raise ValueError(f"Unsupported edge spec length: {edge}")


def _canonical_pattern_edge(edge: Any) -> Tuple[str, str, Optional[Tuple[Any, ...]]]:
    u, v, rel = _parse_edge_spec(edge)
    u_s, v_s = str(u), str(v)
    if u_s <= v_s:
        a, b = u_s, v_s
    else:
        a, b = v_s, u_s
    if rel is None:
        rel_key = None
    elif isinstance(rel, dict):
        rel_allowed = rel.get("rel_in", rel.get("in", []))
        rel_key = tuple(sorted(rel_allowed))
    elif isinstance(rel, (list, tuple, set)):
        rel_key = tuple(sorted(rel))
    else:
        rel_key = (rel,)
    return a, b, rel_key


def _merge_full_motif(constraint: TGD) -> Dict[str, Any]:
    antecedent = constraint.get("antecedent", {})
    consequent = constraint.get("consequent", {})
    nodes: Dict[str, Any] = {}
    for source in (antecedent.get("nodes", {}), consequent.get("nodes", {})):
        for var, spec in source.items():
            if var not in nodes:
                nodes[var] = dict(spec)
                continue
            merged = dict(nodes[var])
            if "in" in merged or "in" in spec:
                merged["in"] = sorted(set(merged.get("in", [])) | set(spec.get("in", [])))
            nodes[var] = merged
    edge_map: Dict[Tuple[str, str, Optional[Tuple[Any, ...]]], Any] = {}
    for edge in list(antecedent.get("edges", [])) + list(consequent.get("edges", [])):
        edge_map[_canonical_pattern_edge(edge)] = edge
    distinct = sorted(set(list(antecedent.get("distinct", [])) + list(consequent.get("distinct", []))))
    return {
        "nodes": nodes,
        "edges": list(edge_map.values()),
        "distinct": distinct,
    }


def _antecedent_only_edges(constraint: TGD) -> List[Any]:
    antecedent_edges = list(constraint.get("antecedent", {}).get("edges", []))
    consequent_keys = {_canonical_pattern_edge(edge) for edge in list(constraint.get("consequent", {}).get("edges", []))}
    return [edge for edge in antecedent_edges if _canonical_pattern_edge(edge) not in consequent_keys]


def _build_edge_bucket(edge_index: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
    bucket: Dict[Tuple[int, int], List[int]] = {}
    src, dst = edge_index[0], edge_index[1]
    for edge_id in range(edge_index.size(1)):
        key = _as_undirected(int(src[edge_id]), int(dst[edge_id]))
        bucket.setdefault(key, []).append(edge_id)
    return bucket


def _drop_edges(edge_index: torch.Tensor, drop_keys: Set[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not drop_keys:
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, keep_mask
    bucket = _build_edge_bucket(edge_index)
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    for key in drop_keys:
        for edge_id in bucket.get(key, []):
            keep_mask[edge_id] = False
    return edge_index[:, keep_mask], keep_mask


def mask_edges_by_constraints(
    data: Data,
    constraints: List[TGD],
    max_masks: int = 1,
    mask_ratio: float | None = None,
    seed: int | None = None,
    prefer_larger_antecedents: bool = True,
    preserve_connectivity: bool = True,
) -> Tuple[Data, List[Tuple[int, int]]]:
    """
    Build an observed graph by matching the full clean motif P ∪ c and removing
    only antecedent-only edges P \\ c under those complete bindings.
    """
    if seed is not None:
        random.seed(seed)

    if mask_ratio is not None:
        total_undirected_edges = data.edge_index.size(1) // 2
        if mask_ratio == 0.0:
            max_masks = 0
        else:
            max_masks = max(1, int(total_undirected_edges * mask_ratio))

    if max_masks == 0:
        return data, []

    weighted_pool: List[Tuple[Tuple[int, int], int]] = []

    for constraint in constraints:
        antecedent_edges = list(constraint.get("antecedent", {}).get("edges", []))
        consequent_edges = list(constraint.get("consequent", {}).get("edges", []))
        droppable_edges = _antecedent_only_edges(constraint)
        if not consequent_edges or not antecedent_edges or not droppable_edges:
            continue

        priority = len(antecedent_edges) if prefer_larger_antecedents else 1
        try:
            full_matches = find_pattern_matches(data, _merge_full_motif(constraint))
        except Exception:
            full_matches = []

        for binding in full_matches:
            for edge_spec in droppable_edges:
                u_var, v_var, _ = _parse_edge_spec(edge_spec)
                if u_var not in binding or v_var not in binding:
                    continue
                u = int(binding[u_var])
                v = int(binding[v_var])
                weighted_pool.append((_as_undirected(u, v), priority))

    if not weighted_pool:
        return data, []

    unique_priorities: Dict[Tuple[int, int], int] = {}
    for key, priority in weighted_pool:
        if key not in unique_priorities or priority > unique_priorities[key]:
            unique_priorities[key] = priority

    ranked = sorted(unique_priorities.items(), key=lambda kv: (kv[1], random.random()), reverse=True)

    if preserve_connectivity:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = int(data.x.size(0))
        elif hasattr(data, "num_nodes") and data.num_nodes is not None:
            num_nodes = int(data.num_nodes)
        else:
            num_nodes = int(data.edge_index.max().item() + 1)
        to_drop = select_edges_preserving_connectivity(
            data.edge_index,
            num_nodes,
            ranked,
            max_masks,
            verbose=True,
        )
    else:
        to_drop = [item[0] for item in ranked[:max_masks]]

    new_edge_index, keep_mask = _drop_edges(data.edge_index, set(to_drop))
    new_data = Data(x=data.x, edge_index=new_edge_index)
    for attr in ("y", "batch"):
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))
    if hasattr(data, "edge_rel_type") and getattr(data, "edge_rel_type") is not None:
        new_data.edge_rel_type = data.edge_rel_type[keep_mask]
    if hasattr(data, "y_type") and getattr(data, "y_type") is not None:
        new_data.y_type = data.y_type
    if hasattr(data, "node_labels") and getattr(data, "node_labels") is not None:
        new_data.node_labels = data.node_labels
    return new_data, to_drop


def mask_edges_for_node_classification(
    data: Data,
    target_node: int,
    constraints: List[TGD],
    num_hops: int = 2,
    max_masks: int = 1,
    mask_ratio: float | None = None,
    seed: int | None = None,
    prefer_larger_antecedents: bool = True,
    preserve_connectivity: bool = True,
) -> Tuple[Data, List[Tuple[int, int]], torch.Tensor]:
    from torch_geometric.utils import k_hop_subgraph

    node_subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    subgraph = Data(
        x=data.x[node_subset],
        edge_index=edge_index_sub,
        num_nodes=int(node_subset.numel()),
    )
    if hasattr(data, "y"):
        subgraph.y = data.y[node_subset]
    if hasattr(data, "y_type") and data.y_type is not None:
        subgraph.y_type = data.y_type[node_subset]
    if hasattr(data, "node_labels") and data.node_labels is not None:
        subgraph.node_labels = data.node_labels[node_subset]
    if hasattr(data, "edge_rel_type") and data.edge_rel_type is not None:
        subgraph.edge_rel_type = data.edge_rel_type[edge_mask]
    if hasattr(data, "batch"):
        subgraph.batch = torch.zeros(subgraph.num_nodes, dtype=torch.long)

    masked_subgraph, dropped_edges = mask_edges_by_constraints(
        subgraph,
        constraints,
        max_masks=max_masks,
        mask_ratio=mask_ratio,
        seed=seed,
        prefer_larger_antecedents=prefer_larger_antecedents,
        preserve_connectivity=preserve_connectivity,
    )
    masked_subgraph.target_node_subgraph_id = int(mapping.item())
    masked_subgraph._nodes_in_full = node_subset.clone()
    masked_subgraph._nodes_in_observed = torch.arange(int(node_subset.numel()))
    masked_subgraph.task = "node"
    return masked_subgraph, dropped_edges, node_subset
