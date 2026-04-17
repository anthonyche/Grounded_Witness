from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import heapq
import time

from itertools import count

# Global counter for heap tiebreaking
_HEAP_SEQ = count()

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected

# --- Safe NetworkX import ---
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

# --- Cosine similarity helper ---
def _cos_sim(a: Tensor, b: Tensor) -> float:
    if a is None or b is None:
        return 0.0
    denom = (a.norm(p=2) * b.norm(p=2)).item()
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _canonical_pair(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)

# Try multiple import paths so this works whether the module is imported as
# `src.apxchase` or plain `apxchase`.
try:
    from constraints import get_constraints  # optional
    from matcher import backchase_repair_cost, find_pattern_matches, MatchResult
    from grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective
    from apxchase import (
        _default_verify_witness as _shared_default_verify_witness,
        _generate_candidate_edge_masks as _shared_generate_candidate_edge_masks,
        _iter_candidate_edge_masks_streaming_fast as _shared_iter_candidate_edge_masks_streaming_fast,
    )
except ImportError:
    from .constraints import get_constraints  # optional
    from .matcher import backchase_repair_cost, find_pattern_matches, MatchResult
    from .grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective
    from .apxchase import (
        _default_verify_witness as _shared_default_verify_witness,
        _generate_candidate_edge_masks as _shared_generate_candidate_edge_masks,
        _iter_candidate_edge_masks_streaming_fast as _shared_iter_candidate_edge_masks_streaming_fast,
    )


def _constraint_names(constraints) -> List[str]:
    #仅用于debug输出,输出constraint的名字
    names = []
    for c in constraints:
        try:
            if isinstance(c, dict) and 'name' in c:
                names.append(str(c['name']))
            elif hasattr(c, 'name'):
                names.append(str(getattr(c, 'name')))
            else:
                names.append(str(c))
        except Exception:
            names.append(str(c))
    return names

# ----------------------------- Helper dataclasses -----------------------------

@dataclass
class WindowEntry:
    score: float
    Gs: Data  # candidate subgraph
    # heapq in python is a min-heap based on the first tuple field (score)
    def as_tuple(self):
        return (self.score, next(_HEAP_SEQ), self.Gs)

# ------------------------------- Default hooks --------------------------------

def _default_verify_witness(model, v_t: Optional[int], Gs: Data) -> bool:
    return _shared_default_verify_witness(model, v_t, Gs)


def _full_motif_from_constraint(constraint) -> Optional[dict]:
    if not isinstance(constraint, dict):
        return None
    def _freeze(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(_freeze(v) for v in obj)
        return obj
    antecedent = constraint.get('antecedent', {})
    consequent = constraint.get('consequent', {})
    a_nodes = list(antecedent.get('nodes', []) or [])
    c_nodes = list(consequent.get('nodes', []) or [])
    if not a_nodes and not c_nodes:
        return None
    nodes = []
    seen_nodes = set()
    for node in a_nodes + c_nodes:
        key = _freeze(node)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        nodes.append(dict(node) if isinstance(node, dict) else node)
    edges = []
    seen_edges = set()
    for edge in list(antecedent.get('edges', []) or []) + list(consequent.get('edges', []) or []):
        key = _freeze(edge)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(dict(edge) if isinstance(edge, dict) else edge)
    return {'nodes': nodes, 'edges': edges}


def _attach_direct_grounding_metadata(Gs: Data, grounded_names: Iterable[str]) -> Set[str]:
    grounded_set = set(grounded_names)
    setattr(Gs, '_grounded_names_set', grounded_set)
    setattr(Gs, 'grounded_names', sorted(grounded_set))
    setattr(Gs, 'grounded', sorted(grounded_set))
    setattr(Gs, 'grounded_constraints', sorted(grounded_set))
    setattr(Gs, 'covered_constraints', sorted(grounded_set))
    setattr(Gs, '_rep_sum', 0.0)
    setattr(Gs, 'rep_sum', 0.0)
    return grounded_set


def _default_conc(Gs: Data) -> float:
    """Conciseness proxy as defined in the paper:
    conc(Gs) = 1 - |E(Gs)| / |E_base|
    Falls back to 1/(1+|E(Gs)|) if E_base is missing or zero.
    """
    m = Gs.edge_index.size(1) if Gs.edge_index.numel() > 0 else 0
    M = getattr(Gs, 'E_base', None)
    if M is None or M == 0:
        return 1.0 / (1 + m)
    return max(0.0, 1.0 - m / float(M))


def _default_rpr(Gs: Data) -> float:
    return float(getattr(Gs, '_alignment', 0.0))

# ------------------------------ Utility methods ------------------------------

def _induce_subgraph_from_edges(H: Data, edge_mask: Tensor) -> Data:
    """
    Build a PyG Data subgraph induced by the edges with mask==True.
    Keeps node features for nodes touched by kept edges and (if isolated) the target node H.root.
    Attaches references to the full graph and chosen edge indices for counterfactual verification.
    """
    ei = H.edge_index
    kept_ei = ei[:, edge_mask]
    H_num_nodes = int(H.num_nodes if getattr(H, 'num_nodes', None) is not None else H.x.size(0))
    
    if kept_ei.numel() > 0:
        nodes = torch.unique(kept_ei.flatten()).to(torch.long)
    else:
        # No edges yet: keep only the root if provided; otherwise keep node 0.
        root_idx = getattr(H, 'root', None)
        if root_idx is None:
            nodes = torch.tensor([0], dtype=torch.long, device=ei.device)
        else:
            # Ensure root_idx is within valid range
            root_idx = int(root_idx)
            if root_idx >= H_num_nodes:
                # Root is out of bounds, use 0 instead
                nodes = torch.tensor([0], dtype=torch.long, device=ei.device)
            else:
                nodes = torch.tensor([root_idx], dtype=torch.long, device=ei.device)

    # Build a compact mapping: original node id -> [0..num_nodes-1] in the candidate
    # Ensure mapping size accommodates all node indices in 'nodes'
    max_node_id = max(int(nodes.max().item()), H_num_nodes - 1)
    mapping = -torch.ones(max_node_id + 1, dtype=torch.long, device=ei.device)
    
    # Filter nodes to be within valid range
    valid_nodes = nodes[nodes < H_num_nodes]
    if valid_nodes.numel() == 0:
        # No valid nodes, use node 0
        valid_nodes = torch.tensor([0], dtype=torch.long, device=ei.device)
    
    mapping[valid_nodes] = torch.arange(valid_nodes.numel(), device=ei.device, dtype=torch.long)
    nodes = valid_nodes

    # Keep only the **selected** edges and relabel their endpoints according to the mapping
    kept_ei = ei[:, edge_mask]
    if kept_ei.numel() > 0:
        u_mapped = mapping[kept_ei[0]]
        v_mapped = mapping[kept_ei[1]]
        relabeled_ei = torch.stack([u_mapped, v_mapped], dim=0)
    else:
        relabeled_ei = torch.empty((2, 0), dtype=torch.long, device=ei.device)

    x = H.x[nodes] if getattr(H, 'x', None) is not None else None
    data = Data(x=x, edge_index=relabeled_ei)
    try:
        assert data.edge_index.size(1) == int(edge_mask.sum().item())
    except Exception:
        pass
    data.num_nodes = int(nodes.numel())
    # carry over batch and task markers if present
    if hasattr(H, 'batch') and H.batch is not None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=nodes.device)
    if hasattr(H, 'task'):
        data.task = H.task
    
    # Extract y_ref, y, and y_type based on task type
    # For NODE classification: index by selected nodes (like x)
    # For GRAPH classification: keep as-is (single value per graph)
    # Detect node task: H.root exists and is valid, OR H.task == 'node', OR y_ref has same length as num_nodes
    root_val = getattr(H, 'root', None)
    task_type = getattr(H, 'task', None)
    y_ref = getattr(H, 'y_ref', None)
    is_node_task = (root_val is not None and root_val >= 0) or \
                   (task_type == 'node') or \
                   (y_ref is not None and y_ref.numel() > 1 and y_ref.size(0) == H.num_nodes)
    
    if hasattr(H, 'y_ref') and H.y_ref is not None:
        if is_node_task:
            data.y_ref = H.y_ref[nodes]  # Node classification: index by nodes
        else:
            data.y_ref = H.y_ref  # Graph classification: keep as-is
    
    if hasattr(H, 'y') and H.y is not None:
        if is_node_task:
            data.y = H.y[nodes]  # Node classification: index by nodes
        else:
            data.y = H.y  # Graph classification: keep as-is
    
    if hasattr(H, 'y_type') and H.y_type is not None:
        if is_node_task:
            data.y_type = H.y_type[nodes]  # Node classification: index by nodes
        else:
            data.y_type = H.y_type  # Graph classification: keep as-is
    if hasattr(H, 'node_labels') and H.node_labels is not None:
        if is_node_task:
            data.node_labels = H.node_labels[nodes]
        else:
            data.node_labels = H.node_labels
    if hasattr(H, 'edge_rel_type') and H.edge_rel_type is not None:
        data.edge_rel_type = H.edge_rel_type[edge_mask].clone()
    
    data.root = getattr(H, 'root', None)
    data.E_base = getattr(H, 'E_base', None)
    
    # FIX: recompute _target_node_subgraph_id in candidate subgraph
    root_val = getattr(data, 'root', None)
    if root_val is not None and root_val >= 0:
        # Ensure root_val is within mapping bounds
        root_idx = int(root_val)
        if root_idx < len(mapping) and mapping[root_idx] >= 0:
            data._target_node_subgraph_id = int(mapping[root_idx].item())
        else:
            # Root not in this subgraph, use 0
            data._target_node_subgraph_id = 0
    elif hasattr(H, '_target_node_subgraph_id'):
        # If H already has _target_node_subgraph_id (from L-hop extraction), remap it
        old_id = H._target_node_subgraph_id
        if old_id < len(mapping) and mapping[old_id] >= 0:
            data._target_node_subgraph_id = int(mapping[old_id].item())
        else:
            data._target_node_subgraph_id = 0
    else:
        data._target_node_subgraph_id = 0
    # Attach references for counterfactual verification:
    # _H_full: the full (masked) graph; _edge_idx_in_full: indices of this candidate's edges in H
    data._H_full = H
    data._edge_idx_in_full = torch.nonzero(edge_mask, as_tuple=False).flatten().clone()
    if hasattr(H, '_nodes_in_observed') and getattr(H, '_nodes_in_observed') is not None:
        data._nodes_in_observed = H._nodes_in_observed[nodes].clone()
    else:
        data._nodes_in_observed = nodes.clone()
    # Persist nodes mapping to full graph for use in repair semantics
    if hasattr(H, '_nodes_in_full') and getattr(H, '_nodes_in_full') is not None:
    # H._nodes_in_full maps H's local ids -> full-graph ids.
    # Our candidate keeps `nodes` (H-local ids), so compose to full ids:
        data._nodes_in_full = H._nodes_in_full[nodes].clone()
    else:
    # Graph-level case (H is a clone of the full graph): ids are already full ids.
        data._nodes_in_full = nodes.clone()
    return data



def _edge_shells_by_hop(H: Data, root: Optional[int], L: int) -> List[Tensor]:
    """Partition edges of H into hop shells E_1..E_L based on min-hop distance
    of their incident nodes from the root. If root is None (graph task),
    return a single shell containing all edges.
    """
    ei = H.edge_index
    if root is None:
        # Graph task: all edges in one shell， no hop distinction
        return [torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)]
    # compute node hops from root on undirected graph
    from collections import deque
    N = H.num_nodes if getattr(H, 'num_nodes', None) is not None else int(H.x.size(0))
    adj = [[] for _ in range(N)]
    und = to_undirected(ei)
    for u, v in und.t().tolist():
        adj[u].append(v)
        adj[v].append(u)
    dist = [-1]*N
    q = deque([root])
    dist[root] = 0
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    # assign edge shell by min hop of its endpoints, clipped to [1,L]
    shells: List[List[int]] = [[] for _ in range(max(L,1))]
    for idx, (u, v) in enumerate(ei.t().tolist()):
        d = min(d if d >= 0 else L for d in (dist[u], dist[v]))
        d = max(1, min(L, d if d > 0 else 1))
        shells[d-1].append(idx)
    return [torch.tensor(s, dtype=torch.long, device=ei.device) for s in shells]

# --- Embedding extraction helper ---
def _extract_node_embeddings(model: torch.nn.Module, H: Data) -> Optional[Tensor]:
    # Try common hooks in order; return Tensor [N, d] or None
    with torch.no_grad():
        try:
            if hasattr(model, 'get_node_embeddings'):
                emb = model.get_node_embeddings(H)
                if isinstance(emb, Tensor):
                    return emb
        except Exception:
            pass
        try:
            # Try to call model appropriately
            model_class_name = model.__class__.__name__
            is_node_model = (hasattr(H, 'task') and H.task == 'node') or \
                           any(name in model_class_name for name in ['GCN_', 'GAT_Yelp', 'SAGE_Yelp', 'GraphSAGE']) or \
                           (not any(word in model_class_name for word in ['Classifier', 'Graph']))
            
            if is_node_model:
                # Node classification: model expects (x, edge_index)
                out = model(H.x, H.edge_index)
            else:
                # Graph classification or unknown: model expects Data object
                out = model(H)
        except Exception:
            out = None
        # Try attributes filled by forward:
        for name in ('last_node_embeddings', 'node_embeddings', 'last_h', 'h'):
            if hasattr(model, name):
                emb = getattr(model, name)
                if (isinstance(emb, Tensor) and emb.dim() == 2 and
                    emb.size(0) == (H.num_nodes if getattr(H, 'num_nodes', None) is not None else (H.x.size(0) if getattr(H, 'x', None) is not None else 0))):
                    return emb
        # Fall back to H.x
        if getattr(H, 'x', None) is not None and isinstance(H.x, Tensor):
            return H.x
    return None


def _build_edmonds_context(H: Data, emb: Optional[Tensor]):
    if not _HAS_NX:
        return None
    import networkx as nx  # local import to keep scope clean

    E = int(H.edge_index.size(1))
    N = int(H.num_nodes if getattr(H, 'num_nodes', None) is not None else (H.x.size(0) if getattr(H, 'x', None) is not None else 0))
    G = nx.DiGraph()
    UG = nx.Graph()
    for n in range(N):
        G.add_node(n)
        UG.add_node(n)

    directed_specs = []
    edge_to_eid = {}
    undir_eids = {}
    undir_weight = {}
    for idx in range(E):
        u = int(H.edge_index[0, idx].item())
        v = int(H.edge_index[1, idx].item())
        w_uv = _cos_sim(emb[u], emb[v]) if emb is not None else 0.0
        w_vu = _cos_sim(emb[v], emb[u]) if emb is not None else 0.0
        G.add_edge(u, v, weight=w_uv, _eid=idx)
        G.add_edge(v, u, weight=w_vu, _eid=idx)
        directed_specs.append((u, v, int(idx), float(w_uv)))
        directed_specs.append((v, u, int(idx), float(w_vu)))
        edge_to_eid[(u, v)] = int(idx)
        edge_to_eid[(v, u)] = int(idx)
        ku, kv = _canonical_pair(u, v)
        undir_eids.setdefault((ku, kv), []).append(int(idx))
        undir_weight[(ku, kv)] = max(float(w_uv), float(w_vu), float(undir_weight.get((ku, kv), float("-inf"))))

    undirected_specs = []
    for (u, v), eids in undir_eids.items():
        base_w = float(undir_weight[(u, v)])
        UG.add_edge(u, v, weight=base_w)
        undirected_specs.append((u, v, tuple(eids), base_w))

    return {
        "graph": G,
        "directed_specs": directed_specs,
        "edge_to_eid": edge_to_eid,
        "undirected_graph": UG,
        "undirected_specs": undirected_specs,
        "undirected_eids": undir_eids,
        "num_edges": E,
    }


# --- Edmonds/Chu–Liu candidate generator ---
def _candidate_by_edmonds(
    H: Data,
    root: Optional[int],
    emb: Optional[Tensor],
    noise_std: float = 1e-3,
    edmonds_ctx=None,
    rng: Optional[torch.Generator] = None,
):
    """
    Returns an edge_mask (Bool[E]) selecting the edges that belong to the
    (maximum) arborescence component reachable from `root`. If NetworkX is
    missing or something fails, falls back to a greedy MST-like heuristic.
    """
    E = H.edge_index.size(1)
    if E == 0:
        return torch.zeros(E, dtype=torch.bool, device=H.edge_index.device)

    device = H.edge_index.device
    edge_mask = torch.zeros(E, dtype=torch.bool, device=device)

    # Build a directed multigraph with both directions per undirected edge.
    if not _HAS_NX:
        # Fallback: select top-degree star around root (or all edges if root None)
        if root is None:
            # keep a spanning forest greedily
            seen = set()
            for idx in range(E):
                u = int(H.edge_index[0, idx].item()); v = int(H.edge_index[1, idx].item())
                if u not in seen or v not in seen:
                    edge_mask[idx] = True
                    seen.add(u); seen.add(v)
            return edge_mask
    try:
        import networkx as nx  # local import to keep scope clean
        ctx = edmonds_ctx or _build_edmonds_context(H, emb)
        if ctx is None:
            raise RuntimeError("NetworkX context unavailable")
        UG = ctx.get("undirected_graph")
        undirected_specs = ctx.get("undirected_specs")
        undirected_eids = ctx.get("undirected_eids", {})

        if rng is None:
            rng = torch.Generator()
            rng.manual_seed(torch.randint(0, 10_000_000, (1,)).item())

        noise_by_edge = None
        if noise_std > 0:
            noise_by_edge = torch.randn(len(undirected_specs), generator=rng).tolist() if undirected_specs is not None else None

        # Edge scores are cosine similarities, which are symmetric by
        # construction. So the bidirected Edmonds objective is equivalent to
        # selecting the maximum-weight underlying undirected tree and then
        # treating its edges as the candidate tree family.
        if UG is None or undirected_specs is None:
            raise RuntimeError("Undirected MST context unavailable")
        for idx_spec, (u, v, _eids, base_w) in enumerate(undirected_specs):
            noise = 0.0 if noise_by_edge is None else float(noise_by_edge[idx_spec]) * noise_std
            UG[u][v]["weight"] = float(base_w + noise)

        T = nx.maximum_spanning_tree(UG, weight='weight')

        selected_nodes = None
        if root is not None and T.number_of_nodes() > 0 and int(root) in T:
            try:
                selected_nodes = set(nx.node_connected_component(T, int(root)))
            except Exception:
                selected_nodes = {int(root)}

        for u, v in T.edges():
            if selected_nodes is not None and (u not in selected_nodes or v not in selected_nodes):
                continue
            ku, kv = _canonical_pair(int(u), int(v))
            for eid in undirected_eids.get((ku, kv), ()):
                edge_mask[int(eid)] = True

        # If still empty (e.g., root isolated), connect greedily by picking incident edges of root
        if edge_mask.sum().item() == 0 and E > 0:
            if root is None:
                return edge_mask
            r = int(root)
            # pick top-k edges adjacent to r (all of them here)
            for idx in range(E):
                u = int(H.edge_index[0, idx].item()); v = int(H.edge_index[1, idx].item())
                if u == r or v == r:
                    edge_mask[idx] = True
        return edge_mask
    except Exception:
        # Robust fallback as above
        if root is None:
            seen = set()
            for idx in range(E):
                u = int(H.edge_index[0, idx].item()); v = int(H.edge_index[1, idx].item())
                if u not in seen or v not in seen:
                    edge_mask[idx] = True
                    seen.add(u); seen.add(v)
        else:
            r = int(root)
            for idx in range(E):
                u = int(H.edge_index[0, idx].item()); v = int(H.edge_index[1, idx].item())
                if u == r or v == r:
                    edge_mask[idx] = True
        return edge_mask


def _seed_nodes_from_mask(H: Data, seed_mask: Tensor, root: Optional[int]) -> Set[int]:
    seed_nodes: Set[int] = set()
    if seed_mask is not None and bool(seed_mask.any().item()):
        kept = H.edge_index[:, seed_mask]
        seed_nodes = set(int(v) for v in torch.unique(kept.flatten()).tolist())
    if not seed_nodes and root is not None:
        seed_nodes.add(int(root))
    return seed_nodes


def _candidate_by_edmonds_around_seed(
    H: Data,
    seed_mask: Tensor,
    root: Optional[int],
    emb: Optional[Tensor],
    edmonds_ctx=None,
) -> Tensor:
    """
    Build a single HeuChase grounding graph G_g by treating the seed candidate
    G_s as a contracted super-node and running one maximum spanning tree on the
    contracted graph. The returned mask always contains the seed edges.
    """
    E = int(H.edge_index.size(1))
    if E == 0:
        return torch.zeros(E, dtype=torch.bool, device=H.edge_index.device)

    seed_mask = seed_mask.clone().to(dtype=torch.bool)
    seed_nodes = _seed_nodes_from_mask(H, seed_mask, root)
    if not seed_nodes:
        return seed_mask
    if not _HAS_NX:
        return seed_mask

    try:
        import networkx as nx  # local import to keep scope clean
        ctx = edmonds_ctx or _build_edmonds_context(H, emb)
        if ctx is None:
            return seed_mask
        undirected_specs = ctx.get("undirected_specs") or []
        super_id = -1
        CG = nx.Graph()
        CG.add_node(super_id)

        bridge_best: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        outside_best: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {}

        for u, v, eids, base_w in undirected_specs:
            in_u = int(u) in seed_nodes
            in_v = int(v) in seed_nodes
            if in_u and in_v:
                continue
            if in_u or in_v:
                other = int(v if in_u else u)
                prev = bridge_best.get(other)
                if prev is None or float(base_w) > prev[0]:
                    bridge_best[other] = (float(base_w), tuple(int(eid) for eid in eids))
                continue
            key = _canonical_pair(int(u), int(v))
            prev = outside_best.get(key)
            if prev is None or float(base_w) > prev[0]:
                outside_best[key] = (float(base_w), tuple(int(eid) for eid in eids))

        for other, (weight, eids) in bridge_best.items():
            CG.add_node(other)
            CG.add_edge(super_id, other, weight=float(weight), eids=eids)
        for (u, v), (weight, eids) in outside_best.items():
            CG.add_node(u)
            CG.add_node(v)
            CG.add_edge(u, v, weight=float(weight), eids=eids)

        if CG.number_of_edges() == 0:
            return seed_mask

        T = nx.maximum_spanning_tree(CG, weight='weight')
        if super_id not in T:
            return seed_mask
        keep_nodes = set(nx.node_connected_component(T, super_id))
        out_mask = seed_mask.clone()
        for u, v, data in T.edges(data=True):
            if u not in keep_nodes or v not in keep_nodes:
                continue
            for eid in data.get('eids', ()):
                out_mask[int(eid)] = True
        return out_mask
    except Exception:
        return seed_mask


def _generate_root_motif_candidates(
    H: Data,
    root: int,
    emb: Optional[Tensor],
    max_candidates: int = 12,
    top_root_branches: int = 6,
    top_extensions: int = 3,
) -> List[Tensor]:
    """
    Generate a small, bounded set of root-centered heuristic candidates.

    HeuC's Edmonds arborescence is useful for selecting a plausible node set,
    but on typed local views it can still be too coarse: either a tiny branch
    or the full component. This helper stays inside HeuC's heuristic family by
    constructing small motif-like candidates anchored at the target root:
      root -- nbr -- ext
    where each undirected pair contributes all corresponding directed edges.
    """
    E = int(H.edge_index.size(1))
    if root is None or E == 0:
        return []

    rel_types = getattr(H, 'edge_rel_type', None)

    def edge_score(u: int, v: int) -> float:
        return _cos_sim(emb[u], emb[v]) if emb is not None else 0.0

    # Group directed edges by unordered node pair + relation type so we can
    # recover a small typed motif while keeping both directions when available.
    pair_groups = {}
    for idx in range(E):
        u = int(H.edge_index[0, idx].item())
        v = int(H.edge_index[1, idx].item())
        rel = int(rel_types[idx].item()) if rel_types is not None else -1
        key = (min(u, v), max(u, v), rel)
        pair_groups.setdefault(key, []).append(idx)

    root_entries = []
    for (u0, v0, rel), idxs in pair_groups.items():
        if root not in (u0, v0):
            continue
        nbr = v0 if u0 == root else u0
        score = edge_score(root, nbr)
        root_entries.append((score, nbr, rel, idxs))
    root_entries.sort(key=lambda x: x[0], reverse=True)

    masks: List[Tensor] = []
    seen = set()
    for _, nbr, root_rel, root_group in root_entries[:max(1, top_root_branches)]:
        ext_entries = []
        for (u0, v0, rel), idxs in pair_groups.items():
            if nbr not in (u0, v0):
                continue
            other = v0 if u0 == nbr else u0
            if other == root:
                continue
            bonus = 0.1 if rel != root_rel else 0.0
            ext_entries.append((edge_score(nbr, other) + bonus, other, rel, idxs))
        ext_entries.sort(key=lambda x: x[0], reverse=True)

        # Candidate 1: one root branch + one extension branch
        for _, _, _, ext_group in ext_entries[:max(1, top_extensions)]:
            mask = torch.zeros(E, dtype=torch.bool, device=H.edge_index.device)
            mask[root_group] = True
            mask[ext_group] = True
            key = tuple(torch.nonzero(mask, as_tuple=False).flatten().tolist())
            if key not in seen:
                masks.append(mask)
                seen.add(key)
                if len(masks) >= max_candidates:
                    return masks

        # Candidate 2: two root branches from the same target, preserving a
        # compact root-centered star when such variation exists.
        for _, nbr2, _, root_group_2 in root_entries[:max(1, top_root_branches)]:
            if nbr2 == nbr:
                continue
            mask = torch.zeros(E, dtype=torch.bool, device=H.edge_index.device)
            mask[root_group] = True
            mask[root_group_2] = True
            key = tuple(torch.nonzero(mask, as_tuple=False).flatten().tolist())
            if key not in seen:
                masks.append(mask)
                seen.add(key)
                if len(masks) >= max_candidates:
                    return masks
    return masks

# --------------------------------- Core class ---------------------------------

class ApxChase:
    def __init__(
        self,
        model: torch.nn.Module,
        Sigma: Optional[Sequence],
        L: int,
        k: int,
        B: int,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 1.0,
        verify_witness_fn: Callable[[torch.nn.Module, Optional[int], Data], bool] = _default_verify_witness,
        gamma_fn: Optional[Callable[[Data, Sequence, int], Set]] = None,
        conc_fn: Callable[[Data], float] = _default_conc,
        rpr_fn: Callable[[Data], float] = _default_rpr,
        seed_per_constraint: int = 2,
        candidate_expand_steps: int = 2,
        candidate_branch_factor: int = 3,
        candidate_beam_width: int = 6,
        candidate_max_masks: int = 48,
        legacy_prefix_checkpoints: int = 6,
        use_ranked_candidate_prioritization: bool = True,
        use_task_aware_hybrid_generation: bool = False,
        ranking_pool_factor: int = 3,
        ranking_diversity_bonus: float = 0.20,
        max_near_full_candidates: int = 16,
        near_full_delete_budget: int = 3,
        near_full_branch_factor: int = 6,
        near_full_beam_width: int = 4,
        large_graph_fast_mode: bool = True,
        large_graph_node_threshold: int = 2500,
        large_graph_edge_threshold: int = 8000,
        large_graph_seed_per_constraint: int = 4,
        large_graph_candidate_expand_steps: int = 1,
        large_graph_candidate_branch_factor: int = 2,
        large_graph_candidate_beam_width: int = 4,
        large_graph_candidate_max_masks: int = 16,
        large_graph_max_consequent_matches_per_constraint: int = 8,
        large_graph_stream_max_candidates: int = 128,
        large_graph_disable_ranked_prioritization: bool = True,
        large_graph_disable_task_aware_hybrid: bool = True,
        large_graph_disable_full_mask_prune: bool = True,
        large_graph_disable_legacy_prefix: bool = True,
        debug: bool = False,
    ):
        self.model = model
        self.Sigma = Sigma
        if self.Sigma is None:
            self.Sigma = []
        self.L = L # L hop subgraph
        self.k = k # window size
        self.B = B # budget for backchase
        self.alpha = alpha # conc weight
        self.beta = beta # rpr weight
        self.gamma = gamma # coverage weight
        self.verify_witness_fn = verify_witness_fn 
        self.conc_fn = conc_fn # conciseness
        self.rpr_fn = rpr_fn # repair penalty
        self.seed_per_constraint = int(seed_per_constraint)
        self.candidate_expand_steps = int(candidate_expand_steps)
        self.candidate_branch_factor = int(candidate_branch_factor)
        self.candidate_beam_width = int(candidate_beam_width)
        self.candidate_max_masks = int(candidate_max_masks)
        self.legacy_prefix_checkpoints = int(legacy_prefix_checkpoints)
        self.use_ranked_candidate_prioritization = bool(use_ranked_candidate_prioritization)
        self.use_task_aware_hybrid_generation = bool(use_task_aware_hybrid_generation)
        self.ranking_pool_factor = int(ranking_pool_factor)
        self.ranking_diversity_bonus = float(ranking_diversity_bonus)
        self.max_near_full_candidates = int(max_near_full_candidates)
        self.near_full_delete_budget = int(near_full_delete_budget)
        self.near_full_branch_factor = int(near_full_branch_factor)
        self.near_full_beam_width = int(near_full_beam_width)
        # HeuChase itself does not use the Apx/Exh large-graph candidate path,
        # but keep the constructor signature aligned with the shared caller.
        self.large_graph_fast_mode = bool(large_graph_fast_mode)
        self.large_graph_node_threshold = int(large_graph_node_threshold)
        self.large_graph_edge_threshold = int(large_graph_edge_threshold)
        self.large_graph_seed_per_constraint = int(large_graph_seed_per_constraint)
        self.large_graph_candidate_expand_steps = int(large_graph_candidate_expand_steps)
        self.large_graph_candidate_branch_factor = int(large_graph_candidate_branch_factor)
        self.large_graph_candidate_beam_width = int(large_graph_candidate_beam_width)
        self.large_graph_candidate_max_masks = int(large_graph_candidate_max_masks)
        self.large_graph_max_consequent_matches_per_constraint = int(large_graph_max_consequent_matches_per_constraint)
        self.large_graph_stream_max_candidates = int(large_graph_stream_max_candidates)
        self.large_graph_disable_ranked_prioritization = bool(large_graph_disable_ranked_prioritization)
        self.large_graph_disable_task_aware_hybrid = bool(large_graph_disable_task_aware_hybrid)
        self.large_graph_disable_full_mask_prune = bool(large_graph_disable_full_mask_prune)
        self.large_graph_disable_legacy_prefix = bool(large_graph_disable_legacy_prefix)
        self.debug = debug
        self._last_run_stats = {}
        self._last_window_update_stats = {}
        # If user did not pass a custom gamma_fn, upgrade to a version
        # that also computes repair costs using backchase on a clean graph.
        if gamma_fn is None:
            self.gamma_fn = self._gamma_with_repair
        else:
            self.gamma_fn = gamma_fn

    def _log(self, msg: str):
        # 输出调试信息
        if self.debug:
            print(f"[ApxChase][DEBUG] {msg}")
    # -------------------------------- Main method --------------------------------

    def _gamma_with_repair(self, Gs: Data, Sigma: Sequence, B: int) -> Set[str]:
        if find_pattern_matches is None or Sigma is None or backchase_repair_cost is None:
            return set()
        G_observed: Data = getattr(self, '_H_observed', None) or Gs
        return evaluate_grounding(Gs, Sigma, B, G_observed, find_pattern_matches, backchase_repair_cost)

    def _candidate_generation_plan(self, H: Data):
        input_num_nodes = int(getattr(H, 'num_nodes', 0) or 0)
        input_num_edges = int(H.edge_index.size(1)) if getattr(H, 'edge_index', None) is not None else 0
        large_graph_fastpath = bool(
            self.large_graph_fast_mode
            and (
                input_num_nodes >= self.large_graph_node_threshold
                or input_num_edges >= self.large_graph_edge_threshold
            )
        )
        kwargs = {
            'seed_per_constraint': self.seed_per_constraint,
            'expand_steps': self.candidate_expand_steps,
            'branch_factor': self.candidate_branch_factor,
            'beam_width': self.candidate_beam_width,
            'max_candidates': self.candidate_max_masks,
            'legacy_checkpoints': self.legacy_prefix_checkpoints,
            'local_budget': self.B,
            'use_ranked_candidate_prioritization': self.use_ranked_candidate_prioritization,
            'use_task_aware_hybrid_generation': self.use_task_aware_hybrid_generation,
            'ranking_pool_factor': self.ranking_pool_factor,
            'diversity_bonus': self.ranking_diversity_bonus,
            'max_near_full_candidates': self.max_near_full_candidates,
            'near_full_delete_budget': self.near_full_delete_budget,
            'near_full_branch_factor': self.near_full_branch_factor,
            'near_full_beam_width': self.near_full_beam_width,
            'max_consequent_matches_per_constraint': None,
            'use_full_mask_prune': True,
            'use_legacy_prefix_masks': True,
            'stream_mode': False,
        }
        mode = "graph_ranked"
        if large_graph_fastpath:
            kwargs['seed_per_constraint'] = min(kwargs['seed_per_constraint'], self.large_graph_seed_per_constraint)
            kwargs['expand_steps'] = min(kwargs['expand_steps'], self.large_graph_candidate_expand_steps)
            kwargs['branch_factor'] = min(kwargs['branch_factor'], self.large_graph_candidate_branch_factor)
            kwargs['beam_width'] = min(kwargs['beam_width'], self.large_graph_candidate_beam_width)
            kwargs['max_candidates'] = min(kwargs['max_candidates'], self.large_graph_candidate_max_masks)
            kwargs['max_consequent_matches_per_constraint'] = self.large_graph_max_consequent_matches_per_constraint
            if self.large_graph_disable_ranked_prioritization:
                kwargs['use_ranked_candidate_prioritization'] = False
            if self.large_graph_disable_task_aware_hybrid:
                kwargs['use_task_aware_hybrid_generation'] = False
            if self.large_graph_disable_full_mask_prune:
                kwargs['use_full_mask_prune'] = False
            if self.large_graph_disable_legacy_prefix:
                kwargs['use_legacy_prefix_masks'] = False
            kwargs['stream_mode'] = True
            kwargs['stream_max_candidates'] = self.large_graph_stream_max_candidates
            mode = "graph_stream_fast"
        return kwargs, mode, large_graph_fastpath

    # ---------------------------- Public entry points ----------------------------
    def explain_node(self, data: Data, v_t: int) -> Tuple[Set, List[Data]]:
        """Run ApxChase for a single target node v_t on PyG Data.
        Returns (Sigma*, S_k).
        """
        # Check if data is already a prepared subgraph (has target_node_subgraph_id)
        if hasattr(data, 'target_node_subgraph_id'):
            # Data is already a prepared subgraph, use it directly
            H = data.clone()
        else:
            # Need to prepare subgraph
            H = self._prepare_subgraph(data, v_t)
        
        H.task = 'node'
        H.root = int(v_t)
        self._H_observed = data
        self._log(f"Start explain_node: v_t={v_t}, |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}, L={self.L}, k={self.k}, B={self.B}, |Sigma|={len(self.Sigma)}")
        if self.debug:
            self._log("Matcher not fully available — consequent-only diagnostics may be skipped.")
        Sigma_star, S_k = self._run(H, root=v_t)
        return Sigma_star, S_k

    def explain_graph(self, data: Data) -> Tuple[Set, List[Data]]:
        """Run ApxChase for a graph-level task (e.g., MUTAG). Root is None,
        all edges are processed in a single shell.
        Returns (Sigma*, S_k).
        """
        H = data.clone()
        H.task = 'graph'
        H.root = None
        if getattr(H, 'num_nodes', None) is None and getattr(H, 'x', None) is not None:
            H.num_nodes = H.x.size(0)
        H._nodes_in_full = torch.arange(int(H.num_nodes), device=H.edge_index.device)
        H._nodes_in_observed = torch.arange(int(H.num_nodes), device=H.edge_index.device)
        H.E_base = H.edge_index.size(1)
        # Grounding semantics are defined against the observed input graph.
        self._H_observed = data
        self._log(f"Start explain_graph: |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}, L={self.L}, k={self.k}, B={self.B}, |Sigma|={len(self.Sigma)}")
        if self.debug:
            self._log("Matcher not fully available — consequent-only diagnostics may be skipped.")
        return self._run(H, root=None)

    # ------------------------------ Internal logic ------------------------------
    def _prepare_subgraph(self, data: Data, v_t: int) -> Data:
        """Extract L-hop subgraph around v_t (node task)."""
        node_idx, ei, mapping, edge_mask = k_hop_subgraph(v_t, self.L, data.edge_index, relabel_nodes=True)
        x = data.x[node_idx] if getattr(data, 'x', None) is not None else None
        out = Data(x=x, edge_index=ei)
        out._nodes_in_full = node_idx.clone()
        out._nodes_in_observed = torch.arange(int(node_idx.numel()), device=ei.device)
        out.num_nodes = int(node_idx.numel())
        # Store the target node's ID in the subgraph (after relabeling)
        out._target_node_subgraph_id = int(mapping.item())
        # carry y_ref if provided (for verify_witness default) - extract only subgraph nodes
        if hasattr(data, 'y_ref'):
            out.y_ref = data.y_ref[node_idx]  # Only extract labels for nodes in subgraph
        if hasattr(data, 'y'):
            out.y = data.y[node_idx]
        if hasattr(data, 'y_type'):
            out.y_type = data.y_type[node_idx]
        if hasattr(data, 'node_labels'):
            out.node_labels = data.node_labels[node_idx]
        if hasattr(data, 'edge_rel_type'):
            out.edge_rel_type = data.edge_rel_type[edge_mask]
        if hasattr(data, 'batch'):
            out.batch = torch.zeros(out.num_nodes, dtype=torch.long, device=ei.device)
        out.E_base = out.edge_index.size(1)
        out.root = v_t
        out.task = 'node'
        return out

    def _update_window(self, W_k: List[Tuple[float, Data]], Gs: Data, covered: Set) -> Set:
        update_t0 = time.time()
        H_view = Gs
        if self.debug:
            self._log(f"Candidate view: |V|={H_view.num_nodes}, |E|={H_view.edge_index.size(1)}")
        # Detailed debug: per-constraint consequent-match counts on this candidate view
        if self.debug:
            if find_pattern_matches is None:
                self._log("Consequent scan skipped: matcher.find_pattern_matches is None (import failed).")
            else:
                per_counts = []
                total_hits = 0
                for t in self.Sigma:
                    try:
                        name = t.get('name', 'unnamed') if isinstance(t, dict) else str(t)
                    except Exception:
                        name = str(t)
                    try:
                        cnt = len(find_pattern_matches(H_view, t.get("consequent", {})))
                    except Exception:
                        cnt = -1  # signal error in matcher
                    if cnt >= 0:
                        total_hits += cnt
                    per_counts.append((name, cnt))
                nonzero = [(n, c) for (n, c) in per_counts if c > 0]
                top5 = sorted(nonzero, key=lambda x: -x[1])[:5]
                self._log(f"Consequent matches on candidate: total={total_hits}; top={top5}")
        pair_t0 = time.time()
        pair_score = pair_quality(Gs, self.conc_fn, self.alpha, self.beta)
        pair_sec = time.time() - pair_t0
        gamma_t0 = time.time()
        Gamma_G = self.gamma_fn(H_view, self.Sigma, self.B)
        gamma_sec = time.time() - gamma_t0
        new_cov = Gamma_G - covered
        if self.debug:
            names_all = _constraint_names(Gamma_G)
            names_new = _constraint_names(new_cov)
            self._log(f"Gamma(G)={len(Gamma_G)} (new={len(new_cov)}); names(new)={names_new[:6]}{'...' if len(names_new)>6 else ''}")
        if self.debug:
            self._log(f"Scores: conc={self.conc_fn(Gs):.4f}, aln={getattr(Gs, '_alignment', 0.0):.4f}, q={pair_score:.4f}")

        current_graphs = [entry[2] for entry in W_k]
        total_constraints = len(self.Sigma)

        if len(current_graphs) < self.k:
            if self.debug:
                if len(Gamma_G) == 0:
                    self._log("Admit verified witness: window not full, grounded coverage remains unchanged.")
                else:
                    self._log("Admit verified witness: window not full.")
            trial_graphs = current_graphs + [Gs]
            W_k.clear()
            for graph in trial_graphs:
                heapq.heappush(W_k, WindowEntry(float(getattr(graph, '_pair_quality', 0.0)), graph).as_tuple())
            self._last_window_update_stats = {
                'pair_sec': float(pair_sec),
                'gamma_sec': float(gamma_sec),
                'window_sec': float(time.time() - update_t0 - pair_sec - gamma_sec),
                'update_total_sec': float(time.time() - update_t0),
                'grounded_count': int(len(Gamma_G)),
                'new_covered_count': int(len(new_cov)),
                'window_size_before': int(len(current_graphs)),
                'window_size_after': int(len(trial_graphs)),
                'admitted': True,
                'replaced': False,
            }
            return window_coverage(trial_graphs)

        if len(Gamma_G) == 0:
            self._log("Skip replacement: no grounded constraints on this candidate.")
            self._last_window_update_stats = {
                'pair_sec': float(pair_sec),
                'gamma_sec': float(gamma_sec),
                'window_sec': float(time.time() - update_t0 - pair_sec - gamma_sec),
                'update_total_sec': float(time.time() - update_t0),
                'grounded_count': 0,
                'new_covered_count': 0,
                'window_size_before': int(len(current_graphs)),
                'window_size_after': int(len(current_graphs)),
                'admitted': False,
                'replaced': False,
            }
            return covered

        current_obj = window_objective(current_graphs, total_constraints, self.gamma)
        best_graphs = current_graphs
        best_obj = current_obj
        for idx in range(len(current_graphs)):
            trial_graphs = current_graphs[:idx] + [Gs] + current_graphs[idx + 1:]
            trial_obj = window_objective(trial_graphs, total_constraints, self.gamma)
            if trial_obj > best_obj:
                best_graphs = trial_graphs
                best_obj = trial_obj

        if best_graphs is current_graphs:
            self._log("Skip: candidate does not improve set-level objective.")
            self._last_window_update_stats = {
                'pair_sec': float(pair_sec),
                'gamma_sec': float(gamma_sec),
                'window_sec': float(time.time() - update_t0 - pair_sec - gamma_sec),
                'update_total_sec': float(time.time() - update_t0),
                'grounded_count': int(len(Gamma_G)),
                'new_covered_count': int(len(new_cov)),
                'window_size_before': int(len(current_graphs)),
                'window_size_after': int(len(current_graphs)),
                'admitted': False,
                'replaced': False,
            }
            return covered

        W_k.clear()
        for graph in best_graphs:
            heapq.heappush(W_k, WindowEntry(float(getattr(graph, '_pair_quality', 0.0)), graph).as_tuple())
        self._last_window_update_stats = {
            'pair_sec': float(pair_sec),
            'gamma_sec': float(gamma_sec),
            'window_sec': float(time.time() - update_t0 - pair_sec - gamma_sec),
            'update_total_sec': float(time.time() - update_t0),
            'grounded_count': int(len(Gamma_G)),
            'new_covered_count': int(len(new_cov)),
            'window_size_before': int(len(current_graphs)),
            'window_size_after': int(len(best_graphs)),
            'admitted': True,
            'replaced': True,
        }
        return window_coverage(best_graphs)

    def _run(self, H: Data, root: Optional[int]) -> Tuple[Set, List[Data]]:
        # shells of edge indices
        shells = _edge_shells_by_hop(H, root=root, L=self.L)
        self._log(f"Edge shells: {len(shells)} levels; total edges M={H.edge_index.size(1)}")
        # Store full masked/induced graph for reuse in _update_window
        self._H_full = H
        # state edge mask (on H.edge_index)
        M = H.edge_index.size(1)
        edge_mask = torch.zeros(M, dtype=torch.bool, device=H.edge_index.device)
        current_nodes = torch.tensor([int(root)], dtype=torch.long, device=H.edge_index.device) if root is not None else torch.tensor([], dtype=torch.long, device=H.edge_index.device)
        W_k: List[Tuple[float, Data]] = []
        covered: Set = set()

        n_candidates = 0
        n_verified = 0
        n_admitted = 0

        for shell in shells:
            # iterate edges in this shell
            for e_idx in (shell if shell.dtype != torch.bool else torch.nonzero(shell, as_tuple=False).flatten()):
                # enforce connectivity: only add if at least one endpoint already present
                u, w = H.edge_index[:, e_idx]
                in_u = (current_nodes == int(u)).any()
                in_w = (current_nodes == int(w)).any()
                # Allow free edge insertion for graph-level tasks (root is None),
                # otherwise enforce connectivity w.r.t. currently grown node set.
                if (root is None) or (current_nodes.numel() > 0 and (in_u or in_w)):
                    # spawn new state by inserting this edge
                    edge_mask[e_idx] = True
                    if self.debug:
                        u_i, w_i = int(u), int(w)
                        self._log(f"Candidate #{n_candidates+1}: add edge ({u_i},{w_i}); current |E(G_s)|={edge_mask.sum().item()}")
                    n_candidates += 1
                    Gs = _induce_subgraph_from_edges(H, edge_mask)
                    ok = self.verify_witness_fn(self.model, root, Gs)
                    if self.debug:
                        self._log("  ✓ VerifyWitness=True" if ok else "  ✗ VerifyWitness=False")
                    if ok:
                        n_verified += 1
                        old_covered = covered
                        covered = self._update_window(W_k, Gs, covered)
                        if len(covered) > len(old_covered):
                            n_admitted += 1
                            if self.debug:
                                self._log(f"  → Admitted: coverage |Γ(W_k)|={len(covered)}; heap size={len(W_k)}")
                    current_nodes = torch.unique(torch.cat([current_nodes, torch.tensor([int(u), int(w)], device=current_nodes.device)]))
                # move on; do not revert the insertion (edge-insertion stream)
        if len(W_k) == 0:
            # Only allow the full observed graph as a fallback witness if it
            # itself satisfies the witness definition.
            if self.verify_witness_fn(self.model, root, H):
                covered = self._update_window(W_k, H, covered)

        final_nodes = (W_k[0][2].num_nodes if len(W_k) > 0 else 0)
        self._log(f"Run stats: candidates={n_candidates}, verified={n_verified}, admitted={n_admitted}, final |W_k|={len(W_k)}, |Γ(W_k)|={len(covered)}, final_nodes={final_nodes}")
        if len(W_k) == 0 and self.debug:
            self._log("No candidates admitted. Consider: increase budget B, relax VerifyWitness, or ensure masking preserves consequent matches so backchase can trigger.")

        S_k = [entry[2] for entry in sorted(W_k, key=lambda t: -t[0])]
        Sigma_star = covered
        # Annotate each witness with its grounded constraints (names) and repair sum
        annotated = []
        for Gs in S_k:
            # Run Γ on the witness itself under the standard c -> P backchase semantics.
            grounded_here = self.gamma_fn(Gs, self.Sigma, self.B)
            try:
                names = list(grounded_here)
                rep_val = float(getattr(Gs, '_rep_sum', 0.0))
                # Common attribute names used across callers
                for attr in ('grounded_names', 'grounded', 'grounded_constraints', 'covered_constraints'):
                    setattr(Gs, attr, names)
                for attr in ('rep_sum', '_rep_sum'):
                    setattr(Gs, attr, rep_val)
                if self.debug:
                    self._log(f"Witness grounded ({len(names)}): {names}; rep_sum={rep_val}")
            except Exception:
                pass
            annotated.append(Gs)
        S_k = annotated
        return Sigma_star, S_k


# --- HeuChase: Edmonds-based candidate generator ---
class HeuChase(ApxChase):
    def __init__(self, *args, m: int = 6, noise_std: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = int(m)
        self.noise_std = float(noise_std)
        if kwargs.get('gamma_fn') is None:
            self.gamma_fn = self._gamma_direct_satisfy

    def _gamma_direct_satisfy(self, Gs: Data, Sigma: Sequence, B: int) -> Set[str]:
        del B
        G_eval = getattr(Gs, '_grounding_graph', None) or Gs
        grounded_names: Set[str] = set()
        if find_pattern_matches is None or Sigma is None:
            return _attach_direct_grounding_metadata(Gs, grounded_names)
        for constraint in Sigma:
            motif = _full_motif_from_constraint(constraint)
            if motif is None:
                continue
            try:
                matches = find_pattern_matches(G_eval, motif)
            except Exception:
                matches = []
            if not matches:
                continue
            try:
                grounded_names.add(str(constraint.get('name', str(constraint))))
            except Exception:
                grounded_names.add(str(constraint))
        _attach_direct_grounding_metadata(G_eval, grounded_names)
        return _attach_direct_grounding_metadata(Gs, grounded_names)

    def _run(self, H: Data, root: Optional[int]) -> Tuple[Set, List[Data]]:
        self._H_full = H
        shells = _edge_shells_by_hop(H, root=root, L=self.L)
        candidate_plan, apx_candidate_mode, large_graph_fastpath = self._candidate_generation_plan(H)
        stream_mode = bool(candidate_plan.pop('stream_mode', False))
        stream_max_candidates = int(candidate_plan.pop('stream_max_candidates', 0) or 0)
        W_k: List[Tuple[float, Data]] = []
        covered: Set = set()

        if getattr(H, 'num_nodes', None) is None and getattr(H, 'x', None) is not None:
            H.num_nodes = H.x.size(0)
        if getattr(H, 'E_base', None) is None:
            H.E_base = H.edge_index.size(1)

        target_id = int(getattr(H, 'root', root if root is not None else -1))
        if root is not None and hasattr(H, '_nodes_in_full'):
            try:
                if int(root) < len(H._nodes_in_full):
                    target_id = int(H._nodes_in_full[int(root)].item())
            except Exception:
                pass
        emb_t0 = time.time()
        emb = _extract_node_embeddings(self.model, H)
        emb_sec = time.time() - emb_t0
        ctx_t0 = time.time()
        edmonds_ctx = _build_edmonds_context(H, emb)
        ctx_sec = time.time() - ctx_t0
        cand_mode = f"{apx_candidate_mode}+heuc_arbo"
        print(
            f"[HeuC | Node {target_id}] stage=edmonds_setup "
            f"|V|={H.num_nodes}, |E|={H.edge_index.size(1)}, emb={emb_sec:.4f}s, ctx={ctx_sec:.4f}s, "
            f"cand_mode={cand_mode}, large={large_graph_fastpath}",
            flush=True,
        )

        if stream_mode:
            candidate_iter = _shared_iter_candidate_edge_masks_streaming_fast(
                H,
                root=root,
                shells=shells,
                seed_budget=int(candidate_plan.get('seed_per_constraint', self.seed_per_constraint))
                * max(1, int(candidate_plan.get('branch_factor', self.candidate_branch_factor))),
                max_total_candidates=max(1, stream_max_candidates),
            )
            distinct_candidate_count = 0
        else:
            candidate_masks = _shared_generate_candidate_edge_masks(
                H,
                root=root,
                shells=shells,
                Sigma=self.Sigma,
                **candidate_plan,
            )
            candidate_iter = iter(candidate_masks)
            distinct_candidate_count = int(len(candidate_masks))

        n_candidates = 0
        n_verified = 0
        n_admitted = 0
        fallback_used = False
        for edge_mask in candidate_iter:
            n_candidates += 1
            if stream_mode:
                distinct_candidate_count += 1
            print(
                f"[HeuC | Node {target_id}] cand_ready#{n_candidates} "
                f"|E(Gs)|={int(edge_mask.sum().item())}",
                flush=True,
            )
            Gs = _induce_subgraph_from_edges(H, edge_mask)
            verify_t0 = time.time()
            ok = self.verify_witness_fn(self.model, root, Gs)
            verify_sec = time.time() - verify_t0
            arbo_sec = 0.0
            update_sec = 0.0
            gamma_sec = 0.0
            grounded_count = 0
            admitted = False
            replaced = False
            grounding_edge_count = 0
            if ok:
                n_verified += 1
                arbo_t0 = time.time()
                grounding_mask = _candidate_by_edmonds_around_seed(
                    H,
                    edge_mask,
                    root=root,
                    emb=emb,
                    edmonds_ctx=edmonds_ctx,
                )
                arbo_sec = time.time() - arbo_t0
                grounding_edge_count = int(grounding_mask.sum().item())
                Gg = _induce_subgraph_from_edges(H, grounding_mask)
                setattr(Gs, '_grounding_graph', Gg)
                setattr(Gs, '_grounding_edge_count', grounding_edge_count)
                old_cov = covered
                update_t0 = time.time()
                covered = self._update_window(W_k, Gs, covered)
                update_sec = time.time() - update_t0
                stats = dict(getattr(self, '_last_window_update_stats', {}) or {})
                gamma_sec = float(stats.get('gamma_sec', 0.0))
                grounded_count = int(stats.get('grounded_count', 0))
                admitted = bool(stats.get('admitted', False))
                replaced = bool(stats.get('replaced', False))
                if len(covered) > len(old_cov):
                    n_admitted += 1
            print(
                f"[HeuC | Node {target_id}] cand#{n_candidates} "
                f"|E(Gs)|={int(edge_mask.sum().item())}, |E(Gg)|={grounding_edge_count}, "
                f"verify={verify_sec:.4f}s, arbo={arbo_sec:.4f}s, grounding={gamma_sec:.4f}s, "
                f"window={update_sec:.4f}s, ok={ok}, grounded={grounded_count}, "
                f"admitted={admitted}, replaced={replaced}",
                flush=True,
            )
            if len(covered) >= len(self.Sigma):
                break

        if len(W_k) == 0:
            if self.verify_witness_fn(self.model, root, H):
                fallback_used = True
                covered = self._update_window(W_k, H, covered)

        if self.debug:
            self._log(f"[HeuChase] stats: candidates={n_candidates}, verified={n_verified}, admitted={n_admitted}, final |W_k|={len(W_k)}, |Γ|={len(covered)}")

        S_k = [entry[2] for entry in sorted(W_k, key=lambda t: -t[0])]
        Sigma_star = covered

        # Annotate witnesses with grounded constraints & rep_sum just like parent
        annotated = []
        for Gs in S_k:
            try:
                grounded_here = getattr(Gs, '_grounded_names_set', None)
                if grounded_here is None:
                    grounded_here = self.gamma_fn(Gs, self.Sigma, self.B)
                names = list(grounded_here)
                rep_val = float(getattr(Gs, '_rep_sum', 0.0))
                for attr in ('grounded_names', 'grounded', 'grounded_constraints', 'covered_constraints'):
                    setattr(Gs, attr, names)
                for attr in ('rep_sum', '_rep_sum'):
                    setattr(Gs, attr, rep_val)
            except Exception:
                pass
            annotated.append(Gs)
        full_witness_edges = extract_witness_edges_in_full(H) if hasattr(H, '_nodes_in_full') else set()
        fallback_selected = any(extract_witness_edges_in_full(Gs) == full_witness_edges for Gs in annotated)
        self._last_run_stats = {
            'num_candidates_generated': int(n_candidates),
            'distinct_candidates_generated': int(distinct_candidate_count),
            'num_candidates_verified': int(n_verified),
            'num_candidates_admitted': int(n_admitted),
            'num_selected_witnesses': int(len(annotated)),
            'num_covered_constraints': int(len(Sigma_star)),
            'fallback_used': bool(fallback_used),
            'fallback_selected': bool(fallback_selected),
            'candidate_generation_mode': cand_mode,
            'large_graph_fastpath': bool(large_graph_fastpath),
        }
        return Sigma_star, annotated
