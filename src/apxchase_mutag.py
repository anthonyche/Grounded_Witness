"""
ApxChase-MUTAG: Multi-center streaming edge-insertion chase for MUTAG graphs.

This is a specialized variant for MUTAG case studies that samples multiple
center nodes and grows candidate subgraphs from each center (inside-out),
generating more diverse candidates for witness generation.

Key differences from base ApxChase:
  • Samples N random nodes as centers (default: 10)
  • For each center, grows edges in hop-distance order (inside-out streaming)
  • All candidates from all centers are pooled together for window management
  • This generates more diverse candidates, improving constraint coverage

External hooks (pluggable) — functions with the `_fn` suffix can be overridden by users:
  - verify_witness_fn(model, v_t, data_subgraph) -> bool
  - gamma_fn(data_subgraph, Sigma, B) -> Set[Constraint]  (uses matcher.Gamma)
  - conc_fn(data_subgraph) -> float
  - rpr_fn(data_subgraph) -> float
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple
import heapq

from itertools import count

# Global counter for heap tiebreaking
_HEAP_SEQ = count()

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected
import networkx as nx

# Try multiple import paths so this works whether the module is imported as
# `src.apxchase` or plain `apxchase`.
try:
    from constraints import get_constraints  # optional
    from matcher import Gamma, backchase_repair_cost, find_head_matches, MatchResult
except ImportError:
    from .constraints import get_constraints  # optional
    from .matcher import Gamma, backchase_repair_cost, find_head_matches, MatchResult


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

def _default_verify_witness(model, v_t: Optional[int], Gs: Data, debug=False) -> bool:
    """
    Default verifier supporting both factual and counterfactual checks.
    - Gs is the candidate subgraph.
    - Factual: prediction on Gs matches full-graph reference in Gs.y_ref.
    - Counterfactual: if Gs._H_full and Gs._edge_idx_in_full are present,
      remove Gs's edges from the full graph, and check that the prediction changes.
    Returns True if either factual OR counterfactual passes.
    """
    model.eval()
    with torch.no_grad():
        # --- Factual: prediction unchanged on Gs vs y_ref ---
        # Determine if model expects (x, edge_index) or Data object
        # is_node_model: True if model expects (x, edge_index) for node classification
        #                False if model expects Data object for graph classification
        model_class_name = model.__class__.__name__
        is_node_model = (hasattr(Gs, 'task') and Gs.task == 'node') or \
                       any(name in model_class_name for name in ['GCN_Yelp', 'GAT_Yelp', 'SAGE_Yelp']) or \
                       (not any(word in model_class_name for word in ['Classifier', 'Graph']))
        
        if is_node_model:
            # Node classification: model expects (x, edge_index)
            out = model(Gs.x, Gs.edge_index)
        else:
            # Graph classification: model expects Data object
            out = model(Gs)
        
        factual_ok = False
        if hasattr(Gs, 'task') and Gs.task == 'node' and v_t is not None:
            # For multi-label: check if ANY label matches (or use sigmoid > 0.5)
            # For multi-class: use argmax
            y_ref = getattr(Gs, 'y_ref', None)
            if y_ref is None:
                if debug:
                    print(f"[VerifyDebug] No y_ref found, factual=True by default")
                factual_ok = True
            else:
                target_subgraph_id = getattr(Gs, '_target_node_subgraph_id', 0)
                if debug:
                    print(f"[VerifyDebug] Factual check: target_id={target_subgraph_id}, |V|={Gs.num_nodes}, |E|={Gs.edge_index.size(1)}")
                    print(f"[VerifyDebug] y_ref shape={y_ref.shape}, out shape={out.shape}")
                    print(f"[VerifyDebug] y_ref[{target_subgraph_id}] dtype={y_ref[target_subgraph_id].dtype}, dim={y_ref[target_subgraph_id].dim()}")
                # FIX: Check if y_ref is multi-dimensional (multi-label) or scalar (multi-class)
                # Multi-label: y_ref has shape [num_nodes, num_classes]
                # Multi-class: y_ref has shape [num_nodes] (scalar labels)
                is_multilabel = y_ref.dim() > 1 or (y_ref[target_subgraph_id].dim() > 0 and y_ref[target_subgraph_id].numel() > 1)
                if is_multilabel:
                    # Multi-label: use sigmoid
                    y_hat = (torch.sigmoid(out) > 0.5).float()
                    # For multi-label, check if predictions match (can use hamming or exact)
                    factual_ok = (y_ref[target_subgraph_id] == y_hat[target_subgraph_id]).all()
                    if debug:
                        print(f"[VerifyDebug] Multi-label: y_ref[{target_subgraph_id}]={y_ref[target_subgraph_id]}, y_hat[{target_subgraph_id}]={y_hat[target_subgraph_id]}, match={factual_ok}")
                else:
                    # Multi-class: use argmax
                    y_hat = out.argmax(dim=-1)
                    factual_ok = (y_ref[target_subgraph_id] == y_hat[target_subgraph_id])
                    if debug:
                        print(f"[VerifyDebug] Multi-class: y_ref[{target_subgraph_id}]={y_ref[target_subgraph_id].item()}, y_hat[{target_subgraph_id}]={y_hat[target_subgraph_id].item()}, match={factual_ok}")
        else:
            # Graph classification
            y_hat = out.argmax(dim=-1)
            y_ref = getattr(Gs, 'y_ref', None)
            if y_ref is None:
                factual_ok = True
            else:
                factual_ok = (y_ref[0] == y_hat[0])

        # --- Counterfactual: prediction flips when removing Gs's edges from full graph ---
        counterfactual_ok = False
        # Check if Gs has references for counterfactual check
        H_full = getattr(Gs, '_H_full', None)
        edge_idx_in_full = getattr(Gs, '_edge_idx_in_full', None)
        if H_full is not None and edge_idx_in_full is not None and edge_idx_in_full.numel() > 0:
            # Construct H_minus by removing candidate's edges from the full graph
            H_minus = H_full.clone()
            # Drop the corresponding columns in edge_index
            mask = torch.ones(H_full.edge_index.size(1), dtype=torch.bool, device=H_full.edge_index.device)
            mask[edge_idx_in_full] = False
            H_minus.edge_index = H_full.edge_index[:, mask]
            # Copy over x, batch, y_ref, task, root, E_base as needed
            if getattr(H_full, 'x', None) is not None:
                H_minus.x = H_full.x
            if hasattr(H_full, 'batch'):
                H_minus.batch = H_full.batch
            if hasattr(H_full, 'y_ref'):
                H_minus.y_ref = H_full.y_ref
            if hasattr(H_full, 'task'):
                H_minus.task = H_full.task
            if hasattr(H_full, 'root'):
                H_minus.root = H_full.root
            if hasattr(H_full, 'E_base'):
                H_minus.E_base = H_full.E_base
            # Run prediction on H_minus
            # Use same logic as above to determine model type
            if is_node_model:
                # Node classification: model expects (x, edge_index)
                out_minus = model(H_minus.x, H_minus.edge_index)
            else:
                # Graph classification: model expects Data object
                out_minus = model(H_minus)
            
            if hasattr(Gs, 'task') and Gs.task == 'node' and v_t is not None:
                # FIX: For counterfactual, use H_minus's target ID (same as H), not Gs's remapped ID
                # H_minus and H_full have same nodes; Gs is a subset with remapped IDs
                target_id_in_H = getattr(H_minus, '_target_node_subgraph_id', 
                                        getattr(H_full, '_target_node_subgraph_id', 0))
                target_id_in_Gs = getattr(Gs, '_target_node_subgraph_id', 0)
                # FIX: Check y_ref to determine if multi-label or multi-class
                y_ref = getattr(Gs, 'y_ref', None)
                is_multilabel = (y_ref is not None and 
                                (y_ref.dim() > 1 or (y_ref[target_id_in_Gs].dim() > 0 and y_ref[target_id_in_Gs].numel() > 1)))
                if is_multilabel:
                    # Multi-label: use sigmoid
                    y_hat_minus = (torch.sigmoid(out_minus) > 0.5).float()
                    y_hat_gs = (torch.sigmoid(out) > 0.5).float()
                    # Counterfactual: ANY label flips (compare H_minus[H_id] vs Gs[Gs_id])
                    counterfactual_ok = (y_hat_gs[target_id_in_Gs] != y_hat_minus[target_id_in_H]).any()
                else:
                    # Multi-class: use argmax
                    y_hat_minus = out_minus.argmax(dim=-1)
                    y_hat_gs = out.argmax(dim=-1)
                    counterfactual_ok = (y_hat_gs[target_id_in_Gs] != y_hat_minus[target_id_in_H])
            else:
                # Graph classification
                y_hat_minus = out_minus.argmax(dim=-1)
                y_hat_gs = out.argmax(dim=-1)
                counterfactual_ok = (y_hat_gs[0] != y_hat_minus[0])

        # Accept if either factual OR counterfactual passes
        return factual_ok or counterfactual_ok


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
    """Repair penalty proxy as defined in the paper:
    rpr(Gs) = 1 - sum_rep / (|E(Gs)| + sum_rep)
    where sum_rep = sum of repairs from Gamma(Gs, B).
    """
    #目前这个函数有问题，一会儿来fix，输出持续等于1
    sum_rep = getattr(Gs, '_rep_sum', 0.0)
    m = Gs.edge_index.size(1) if Gs.edge_index.numel() > 0 else 0
    if m == 0 and sum_rep == 0:
        return 0.0
    return 1.0 - (sum_rep / (m + sum_rep))

# ------------------------------ Utility methods ------------------------------

def _graph_signature(data: Data) -> Tuple:
    """
    Compute a canonical signature for a graph to detect duplicates.
    Returns a tuple of (sorted_edges, node_count) for structural comparison.
    """
    if data.edge_index.numel() == 0:
        return (tuple(), int(data.num_nodes))
    
    # Convert to undirected edge set and sort for canonical form
    ei = data.edge_index.cpu().numpy()
    edges = set()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        # Normalize edge direction for undirected comparison
        edges.add((min(u, v), max(u, v)))
    
    return (tuple(sorted(edges)), int(data.num_nodes))


def _is_connected_subgraph(data: Data, root: Optional[int] = None, debug: bool = False) -> bool:
    """
    Check if a subgraph is connected (for graph-level) or contains root in largest component (for node-level).
    
    For graph-level (root=None): Accept if graph has only one connected component.
    For node-level: Accept if root is in the largest connected component.
    """
    num_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0
    
    if num_edges == 0:
        # No edges - only accept if single node (root)
        result = data.num_nodes <= 1
        if debug:
            print(f"[ConnCheck] No edges: |V|={data.num_nodes}, accepting={result}")
        return result
    
    # Build NetworkX graph for connectivity check
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    ei = data.edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        G.add_edge(u, v)
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # Check if all nodes are covered by edges
    nodes_in_edges = set()
    for i in range(ei.shape[1]):
        nodes_in_edges.add(int(ei[0, i]))
        nodes_in_edges.add(int(ei[1, i]))
    isolated_nodes = set(range(data.num_nodes)) - nodes_in_edges
    
    if debug:
        print(f"[ConnCheck] |V|={data.num_nodes}, |E|={num_edges}, components={len(components)}, root={root}")
        print(f"[ConnCheck] Nodes in edges: {len(nodes_in_edges)}, Isolated nodes: {len(isolated_nodes)}")
        if isolated_nodes:
            print(f"[ConnCheck] WARNING: Isolated node IDs: {sorted(isolated_nodes)}")
        if len(components) > 1:
            print(f"[ConnCheck] Component sizes: {[len(c) for c in components]}")
    
    if root is None:
        # Graph-level: require single connected component AND no isolated nodes
        return len(components) == 1 and len(isolated_nodes) == 0
    else:
        # Node-level: require root in largest component
        if not components:
            return False
        largest_component = max(components, key=len)
        # Find root's ID in this subgraph
        root_subgraph_id = getattr(data, '_target_node_subgraph_id', 0)
        return root_subgraph_id in largest_component and len(isolated_nodes) == 0


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
    
    data.root = getattr(H, 'root', None)
    data.E_base = getattr(H, 'E_base', None)
    # FIX: recompute _target_node_subgraph_id in candidate subgraph
    # H.root is the target node ID in H; mapping[H.root] is its new ID in candidate
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
    
    # Validate root index
    if root < 0 or root >= N:
        raise ValueError(f"Root node index {root} is out of bounds for graph with {N} nodes. "
                        f"Expected root in range [0, {N-1}].")
    
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
        debug: bool = False,
        num_centers: int = 10,  # NEW: number of center nodes to sample
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
        self.num_centers = num_centers  # NEW: multi-center sampling
        self.verify_witness_fn = verify_witness_fn 
        self.conc_fn = conc_fn # conciseness
        self.rpr_fn = rpr_fn # repair penalty
        self.debug = debug
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
        """
        STRICT repair semantics (no heuristic completion):
        1) Run HEAD matching **on the candidate** subgraph `Gs` to get bindings (var -> Gs node id).
        2) Map each binding to **full-graph ids** via `Gs._nodes_in_full`.
        3) On the clean/original graph `self._H_clean` (if provided; else fall back to `Gs`),
            call `backchase_repair_cost(clean_graph, tgd, binding_full_ids, B)` to obtain the
            minimal number of BODY edges that must be added for this binding to satisfy BODY.
        4) If the minimal repair cost over all bindings ≤ B, the constraint is grounded.
            Accumulate the minimal repair cost per-constraint into `Gs._rep_sum` for downstream rpr().
        """
        grounded_names: Set[str] = set()
        total_rep = 0

        # Required hooks
        if find_head_matches is None or Sigma is None or backchase_repair_cost is None:
            return set()

        # Clean graph for repair semantics
        G_clean: Data = getattr(self, '_H_clean', None)
        if G_clean is None:
            G_clean = Gs  # conservative fallback

        # Candidate node-id -> full-graph node-id mapping is required
        nodes_in_full = getattr(Gs, '_nodes_in_full', None)
        if nodes_in_full is None:
            return set()
        nodes_in_full = nodes_in_full.tolist()

        def _map_full(i_view: int) -> int:
            return int(nodes_in_full[int(i_view)])

        for tgd in Sigma:
            # Name for logging/return
            try:
                name = tgd.get('name', 'unnamed') if isinstance(tgd, dict) else str(tgd)
            except Exception:
                name = str(tgd)

            # 1) HEAD matches on the candidate
            try:
                matches = find_head_matches(Gs, tgd)
            except Exception:
                matches = []

            best_rep = None
            # 2–3) Evaluate strict repair cost on clean graph for each binding
            for bind_view in matches:
                try:
                    bind_full = {var: _map_full(nv) for var, nv in bind_view.items()}
                except Exception:
                    continue
                try:
                    rep_cost = backchase_repair_cost(G_clean, tgd, bind_full, B)
                except Exception:
                    rep_cost = None

                # Normalize different possible return types to a numeric cost
                # Acceptable forms:
                #   - int/float cost
                #   - (cost, ...) tuple or [cost, ...] list
                #   - { 'cost': cost, ... } dict
                if isinstance(rep_cost, (tuple, list)):
                    rep_cost = rep_cost[0] if len(rep_cost) > 0 else None
                elif isinstance(rep_cost, dict):
                    rep_cost = rep_cost.get('cost', None)

                # Ensure rep_cost is numeric
                if rep_cost is None:
                    continue
                if not isinstance(rep_cost, (int, float)):
                    # Unrecognized form; skip this binding safely
                    continue

                if rep_cost <= B:
                    if best_rep is None or rep_cost < best_rep:
                        best_rep = int(rep_cost)

            if best_rep is not None and best_rep <= B:
                grounded_names.add(name)
                total_rep += best_rep

        # Persist accumulated repair cost for rpr(Gs)
        try:
            setattr(Gs, '_rep_sum', float(total_rep))
        except Exception:
            pass

        return grounded_names

    # ---------------------------- Public entry points ----------------------------
    def explain_node(self, data: Data, v_t: int) -> Tuple[Set, List[Data]]:
        """Run ApxChase for a single target node v_t on PyG Data.
        The input `data` should already be the L-hop subgraph around v_t.
        v_t should be the node's ID within this subgraph (after relabeling).
        Returns (Sigma*, S_k).
        """
        # Use the input data directly (it's already the prepared subgraph)
        H = data.clone()
        H.task = 'node'
        H.root = int(v_t)
        if not hasattr(H, 'num_nodes'):
            H.num_nodes = H.x.size(0) if H.x is not None else 0
        self._H_clean = getattr(data, '_clean', data)
        self._log(f"Start explain_node: v_t={v_t}, |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}, L={self.L}, k={self.k}, B={self.B}, |Sigma|={len(self.Sigma)}")
        if self.debug:
            self._log("Debugging mode — head-only diagnostics may be skipped.")
            # Print actual constraint names and their HEAD edge counts
            constraint_info = []
            for c in self.Sigma:
                name = c.get('name', 'unnamed')
                head_edges = len(c.get('head', {}).get('edges', []))
                constraint_info.append(f"{name}({head_edges}e)")
            self._log(f"Loaded constraints: {constraint_info}")
        return self._run(H, root=v_t)

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
        H.E_base = H.edge_index.size(1)
        # Keep a handle to a clean/original graph for repair-cost evaluation.
        # If the caller attached an attribute `_clean` (unmasked), use it;
        # otherwise fall back to the current masked graph.
        self._H_clean = getattr(data, '_clean', data)
        self._log(f"Start explain_graph on MASKED graph: |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}, L={self.L}, k={self.k}, B={self.B}, |Sigma|={len(self.Sigma)}")
        if self.debug:
            self._log("Matcher not fully available — head-only diagnostics may be skipped.")
        return self._run(H, root=None)

    # ------------------------------ Internal logic ------------------------------
    def _prepare_subgraph(self, data: Data, v_t: int) -> Data:
        """Extract L-hop subgraph around v_t (node task)."""
        node_idx, ei, mapping, _ = k_hop_subgraph(v_t, self.L, data.edge_index, relabel_nodes=True)
        x = data.x[node_idx] if getattr(data, 'x', None) is not None else None
        out = Data(x=x, edge_index=ei)
        out._nodes_in_full = node_idx.clone()
        out.num_nodes = int(node_idx.numel())
        # Store the target node's ID in the subgraph (after relabeling)
        out._target_node_subgraph_id = int(mapping.item())
        # carry y_ref if provided (for verify_witness default) - extract only subgraph nodes
        if hasattr(data, 'y_ref'):
            out.y_ref = data.y_ref[node_idx]  # Only extract labels for nodes in subgraph
        # carry y (true labels) for matcher
        if hasattr(data, 'y'):
            out.y = data.y[node_idx]  # Extract true labels for subgraph nodes
        # carry y_type (KMeans cluster labels) for TGD matching
        if hasattr(data, 'y_type'):
            out.y_type = data.y_type[node_idx]  # Extract type labels for subgraph nodes
        if hasattr(data, 'batch'):
            out.batch = torch.zeros(out.num_nodes, dtype=torch.long, device=ei.device)
        out.E_base = out.edge_index.size(1)
        out.root = v_t  # Store original target node ID (in full graph)
        out.task = 'node'
        return out

    def _update_window(self, W_k: List[Tuple[float, Data]], Gs: Data, covered: Set, seen_signatures: Optional[Set] = None) -> Set:
        """
        Update streaming window with improved policies for case studies.
        
        IMPROVEMENTS:
        1. Deduplication: Skip structurally identical graphs (same edge set)
        2. Connectivity enforcement: Only accept connected subgraphs
        3. Relaxed admission: Any candidate grounding constraints can compete
        4. Score-based ranking: delta = alpha*conc + beta*rpr + gamma*coverage
        
        Returns the updated coverage set Γ(W_k).
        """
        # IMPROVEMENT 1: Check connectivity (reject disconnected subgraphs)
        root_for_check = getattr(self, '_H_full', None)
        root_id = getattr(root_for_check, 'root', None) if root_for_check is not None else None
        
        if not _is_connected_subgraph(Gs, root=root_id, debug=self.debug):
            if self.debug:
                self._log(f"Skip: disconnected subgraph (|V|={Gs.num_nodes}, |E|={Gs.edge_index.size(1)//2})")
            return covered
        
        # IMPROVEMENT 2: Check for duplicate structure
        if seen_signatures is not None:
            sig = _graph_signature(Gs)
            if sig in seen_signatures:
                if self.debug:
                    self._log(f"Skip: duplicate structure detected (|V|={Gs.num_nodes}, |E|={Gs.edge_index.size(1)//2})")
                return covered
            # Mark as seen (will be added to set if admitted)
        
        # Use the candidate itself for head matching / Γ evaluation
        H_view = Gs
        if self.debug:
            self._log(f"Candidate view: |V|={H_view.num_nodes}, |E|={H_view.edge_index.size(1)//2}")
        
        # Detailed debug: per-constraint head-only match counts on this candidate view
        if self.debug:
            if find_head_matches is None:
                self._log("HEAD-scan skipped: matcher.find_head_matches is None (import failed).")
            else:
                per_counts = []
                total_hits = 0
                for t in self.Sigma:
                    try:
                        name = t.get('name', 'unnamed') if isinstance(t, dict) else str(t)
                    except Exception:
                        name = str(t)
                    try:
                        cnt = len(find_head_matches(H_view, t))
                    except Exception:
                        cnt = -1  # signal error in matcher
                    if cnt >= 0:
                        total_hits += cnt
                    per_counts.append((name, cnt))
                nonzero = [(n, c) for (n, c) in per_counts if c > 0]
                top5 = sorted(nonzero, key=lambda x: -x[1])[:5]
                self._log(f"HEAD matches on candidate: total={total_hits}; top={top5}")
        
        # Compute Gamma on candidate itself
        Gamma_G = self.gamma_fn(H_view, self.Sigma, self.B)
        
        if self.debug:
            names_all = _constraint_names(Gamma_G)
            self._log(f"Gamma(G)={len(Gamma_G)}; names={names_all[:6]}{'...' if len(names_all)>6 else ''}")
        
        # Reject if NO constraints are grounded
        if len(Gamma_G) == 0:
            if self.debug:
                self._log("Skip: no grounded constraints on this candidate.")
            return covered
        
        # Compute score based on TOTAL coverage (not just new coverage)
        # This allows candidates with overlapping constraint coverage to compete
        conc = self.conc_fn(Gs)
        rpr = self.rpr_fn(Gs)
        coverage_ratio = len(Gamma_G) / max(1, len(self.Sigma))
        delta = self.alpha * conc + self.beta * rpr + self.gamma * coverage_ratio
        
        if self.debug:
            self._log(f"Scores: conc={conc:.4f}, rpr={rpr:.4f}, coverage={coverage_ratio:.4f} ({len(Gamma_G)}/{len(self.Sigma)}), delta={delta:.4f}")
        
        entry = WindowEntry(delta, Gs)
        
        # Window management: maintain top-k by delta score
        if len(W_k) < self.k:
            # Window not full - always admit
            heapq.heappush(W_k, entry.as_tuple())
            if self.debug:
                self._log(f"Heap push (|W_k| -> {len(W_k)}).")
            # Update global coverage tracking
            covered = covered | Gamma_G
            # Mark signature as seen
            if seen_signatures is not None:
                seen_signatures.add(_graph_signature(Gs))
        else:
            # Window full - compete with worst candidate
            if delta > W_k[0][0]:
                if self.debug:
                    self._log(f"Heap replace: popped min delta={W_k[0][0]:.4f}, pushed delta={delta:.4f}.")
                heapq.heapreplace(W_k, entry.as_tuple())
                # Update global coverage tracking
                covered = covered | Gamma_G
                # Mark signature as seen
                if seen_signatures is not None:
                    seen_signatures.add(_graph_signature(Gs))
            else:
                if self.debug:
                    self._log(f"Skip: delta={delta:.4f} <= heap-min={W_k[0][0]:.4f}.")
        
        return covered

    def _run_multi_center(self, H: Data, root: Optional[int]) -> Tuple[Set, List[Data]]:
        """
        Multi-center BFS-style edge insertion for MUTAG case studies.
        
        Strategy:
        1. Iterate through ALL nodes in the graph as potential centers
        2. For each center, perform BFS-style edge expansion to maintain connectivity
        3. Each expansion step (adding one edge) produces a candidate subgraph
        4. All candidates compete for window admission based on weighted score
        5. Deduplication ensures structurally unique witnesses
        
        This ensures:
        - All explanation subgraphs are connected
        - Comprehensive exploration of the graph structure
        - Diverse witnesses with different constraint patterns
        """
        M = H.edge_index.size(1)
        N = H.num_nodes if getattr(H, 'num_nodes', None) is not None else int(H.x.size(0))
        
        self._log(f"Start BFS multi-center streaming on MASKED graph: |V(H)|={N}, |E(H)|={M}")
        self._log(f"Note: BFS will grow candidates FROM masked graph edges (ensuring connectivity)")
        
        # Store full masked/induced graph for reuse in _update_window
        self._H_full = H
        
        # Use ALL nodes as centers (not random sampling)
        center_nodes = torch.arange(N, device=H.edge_index.device)
        self._log(f"Processing {N} center nodes (all nodes in graph)")
        
        # Global window and coverage tracking
        W_k: List[Tuple[float, Data]] = []
        covered: Set = set()
        seen_signatures: Set = set()  # Track seen graph structures for deduplication
        
        # Statistics
        total_candidates = 0
        total_verified = 0
        total_admitted = 0
        total_rejected_dup = 0
        total_rejected_disconnected = 0
        
        # Build adjacency list for BFS
        import networkx as nx
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(N))
        ei = H.edge_index.cpu().numpy()
        edge_list = []
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            G_nx.add_edge(u, v)
            edge_list.append((u, v, i))  # Store edge index for masking
        
        # Process each center node
        for center_idx, center in enumerate(center_nodes):
            center_id = int(center.item())
            
            # Log progress every 5 centers or for first/last
            if self.debug or center_idx == 0 or center_idx == len(center_nodes) - 1 or center_idx % 5 == 0:
                self._log(f"\n--- Center {center_idx+1}/{N}: node {center_id} ---")
            
            # BFS from this center to order edges by distance
            from collections import deque
            visited_nodes = {center_id}
            visited_edges = set()
            queue = deque([center_id])
            bfs_edge_order = []  # Edges in BFS order
            
            while queue:
                current = queue.popleft()
                # Get all neighbors
                for neighbor in G_nx.neighbors(current):
                    edge_pair = (min(current, neighbor), max(current, neighbor))
                    if edge_pair not in visited_edges:
                        visited_edges.add(edge_pair)
                        # Find edge index in original edge_index
                        for u, v, idx in edge_list:
                            if (min(u, v), max(u, v)) == edge_pair:
                                bfs_edge_order.append(idx)
                                break
                        
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            queue.append(neighbor)
            
            # Now grow subgraph by adding edges in BFS order
            edge_mask = torch.zeros(M, dtype=torch.bool, device=H.edge_index.device)
            n_candidates = 0
            n_verified = 0
            n_admitted = 0
            
            for edge_idx in bfs_edge_order:
                # Add this edge to the mask
                edge_mask[edge_idx] = True
                n_candidates += 1
                total_candidates += 1
                
                if self.debug and n_candidates <= 3:
                    u, v = H.edge_index[:, edge_idx]
                    self._log(f"    Candidate #{n_candidates}: add edge ({int(u)},{int(v)}); |E(G_s)|={edge_mask.sum().item()}")
                
                # Induce subgraph from current edge set
                Gs = _induce_subgraph_from_edges(H, edge_mask)
                
                # Verify witness
                ok = self.verify_witness_fn(self.model, root, Gs)
                
                if self.debug and n_candidates <= 3:
                    self._log("      ✓ VerifyWitness=True" if ok else "      ✗ VerifyWitness=False")
                
                if ok:
                    n_verified += 1
                    total_verified += 1
                    old_covered = covered
                    old_window_size = len(W_k)
                    covered = self._update_window(W_k, Gs, covered, seen_signatures)
                    
                    # Check if admitted to window
                    if len(W_k) > old_window_size or len(covered) > len(old_covered):
                        n_admitted += 1
                        total_admitted += 1
                        if self.debug and n_admitted <= 5:
                            self._log(f"      → Admitted: coverage |Γ(W_k)|={len(covered)}; heap size={len(W_k)}")
                    
                    # Early stopping: if all constraints are grounded, stop
                    if len(covered) >= len(self.Sigma):
                        if self.debug:
                            self._log(f"  Early stop at center {center_idx+1}: all {len(self.Sigma)} constraints grounded!")
                        break
            
            if (self.debug or center_idx % 5 == 0) and n_candidates > 0:
                self._log(f"  Center {center_idx+1} stats: candidates={n_candidates}, verified={n_verified}, admitted={n_admitted}")
            
            # Check early stop condition for center loop
            if len(covered) >= len(self.Sigma):
                self._log(f"All constraints covered after {center_idx+1}/{N} centers, stopping early.")
                break
        
        # Fallback: if no candidates admitted, add H itself
        if len(W_k) == 0:
            self._log("Fallback: no candidates admitted, adding full graph H")
            covered = self._update_window(W_k, H, covered)
        
        final_nodes = (W_k[0][2].num_nodes if len(W_k) > 0 else 0)
        self._log(f"\nFinal stats: total_candidates={total_candidates}, total_verified={total_verified}, total_admitted={total_admitted}, |W_k|={len(W_k)}, |Γ(W_k)|={len(covered)}, unique_structures={len(seen_signatures)}, final_nodes={final_nodes}")
        
        if len(W_k) == 0 and self.debug:
            self._log("No candidates admitted. Consider: increase budget B, relax VerifyWitness, or ensure masking removes head-edges so backchase can trigger.")
        
        # Extract and annotate witnesses
        S_k = [entry[2] for entry in sorted(W_k, key=lambda t: -t[0])]
        Sigma_star = covered
        
        # FINAL VALIDATION: Verify all witnesses are connected
        if self.debug:
            self._log("\n=== Final Witness Connectivity Check ===")
        
        # Annotate each witness with its grounded constraints (names) and repair sum
        annotated = []
        for idx, Gs in enumerate(S_k):
            # Double-check connectivity
            is_conn = _is_connected_subgraph(Gs, root=root, debug=self.debug)
            if not is_conn:
                if self.debug:
                    self._log(f"WARNING: Witness #{idx+1} is DISCONNECTED! |V|={Gs.num_nodes}, |E|={Gs.edge_index.size(1)//2}")
                # Skip disconnected witnesses
                continue
            
            grounded_here = self.gamma_fn(Gs, self.Sigma, self.B)
            try:
                names = list(grounded_here)
                rep_val = float(getattr(Gs, '_rep_sum', 0.0))
                for attr in ('grounded_names', 'grounded', 'grounded_constraints', 'covered_constraints'):
                    setattr(Gs, attr, names)
                for attr in ('rep_sum', '_rep_sum'):
                    setattr(Gs, attr, rep_val)
                if self.debug:
                    self._log(f"Witness #{idx+1} OK: grounded ({len(names)}): {names}; rep_sum={rep_val}")
            except Exception:
                pass
            annotated.append(Gs)
        
        if self.debug:
            self._log(f"=== Kept {len(annotated)}/{len(S_k)} connected witnesses ===\n")
        
        S_k = annotated
        return Sigma_star, S_k

    def _run(self, H: Data, root: Optional[int]) -> Tuple[Set, List[Data]]:
        """
        Main run method that delegates to multi-center streaming for graph tasks.
        For node tasks, falls back to single-center (root-based) streaming.
        """
        # For graph-level tasks (MUTAG), use multi-center streaming
        if root is None or H.task == 'graph':
            return self._run_multi_center(H, root)
        
        # For node-level tasks, use original single-center streaming
        # (original _run code preserved below)
        shells = _edge_shells_by_hop(H, root=root, L=self.L)
        self._log(f"Edge shells: {len(shells)} levels; total edges M={H.edge_index.size(1)}")
        self._H_full = H
        M = H.edge_index.size(1)
        edge_mask = torch.zeros(M, dtype=torch.bool, device=H.edge_index.device)
        current_nodes = torch.tensor([int(root)], dtype=torch.long, device=H.edge_index.device)
        W_k: List[Tuple[float, Data]] = []
        covered: Set = set()
        seen_signatures: Set = set()  # Track seen graph structures for deduplication

        n_candidates = 0
        n_verified = 0
        n_admitted = 0

        for shell in shells:
            for e_idx in (shell if shell.dtype != torch.bool else torch.nonzero(shell, as_tuple=False).flatten()):
                u, w = H.edge_index[:, e_idx]
                in_u = (current_nodes == int(u)).any()
                in_w = (current_nodes == int(w)).any()
                if current_nodes.numel() > 0 and (in_u or in_w):
                    edge_mask[e_idx] = True
                    if self.debug:
                        u_i, w_i = int(u), int(w)
                        self._log(f"Candidate #{n_candidates+1}: add edge ({u_i},{w_i}); current |E(G_s)|={edge_mask.sum().item()}")
                    n_candidates += 1
                    Gs = _induce_subgraph_from_edges(H, edge_mask)
                    if self.debug and n_candidates == 1:
                        ok = _default_verify_witness(self.model, root, Gs, debug=True)
                    else:
                        ok = self.verify_witness_fn(self.model, root, Gs)
                    if self.debug:
                        self._log("  ✓ VerifyWitness=True" if ok else "  ✗ VerifyWitness=False")
                    if ok:
                        n_verified += 1
                        old_covered = covered
                        old_window_size = len(W_k)
                        covered = self._update_window(W_k, Gs, covered, seen_signatures)
                        if len(W_k) > old_window_size or len(covered) > len(old_covered):
                            n_admitted += 1
                            if self.debug:
                                self._log(f"  → Admitted: coverage |Γ(W_k)|={len(covered)}; heap size={len(W_k)}")
                        if len(covered) >= len(self.Sigma):
                            if self.debug:
                                self._log(f"Early stop: all {len(self.Sigma)} constraints grounded!")
                            break
                    current_nodes = torch.unique(torch.cat([current_nodes, torch.tensor([int(u), int(w)], device=current_nodes.device)]))
            if len(covered) >= len(self.Sigma):
                break
        if len(W_k) == 0:
            covered = self._update_window(W_k, H, covered, seen_signatures)

        final_nodes = (W_k[0][2].num_nodes if len(W_k) > 0 else 0)
        self._log(f"Run stats: candidates={n_candidates}, verified={n_verified}, admitted={n_admitted}, final |W_k|={len(W_k)}, |Γ(W_k)|={len(covered)}, unique_structures={len(seen_signatures)}, final_nodes={final_nodes}")
        if len(W_k) == 0 and self.debug:
            self._log("No candidates admitted. Consider: increase budget B, relax VerifyWitness, or ensure masking removes head-edges so backchase can trigger.")

        S_k = [entry[2] for entry in sorted(W_k, key=lambda t: -t[0])]
        Sigma_star = covered
        annotated = []
        for Gs in S_k:
            grounded_here = self.gamma_fn(Gs, self.Sigma, self.B)
            try:
                names = list(grounded_here)
                rep_val = float(getattr(Gs, '_rep_sum', 0.0))
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