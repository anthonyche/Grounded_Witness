"""
ExhaustChase: Exhaustive chase-based baseline for witness generation.

Key difference from ApxChase:
  1) First, exhaustively enforce ALL TGD rules until no violations remain (clean graph).
  2) Then, perform candidate generation using the same method as ApxChase.
  
This baseline is expected to be slower due to the exhaustive enforcement overhead,
which is intentionally included in timing measurements to demonstrate the cost.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import heapq
import time

from itertools import count

# Global counter for heap tiebreaking
_HEAP_SEQ = count()

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected

# Try multiple import paths so this works whether the module is imported as
# `src.apxchase` or plain `apxchase`.
try:
    from constraints import get_constraints  # optional
    from matcher import backchase_repair_cost, find_pattern_matches, MatchResult
    from grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective
    from apxchase import _generate_candidate_edge_masks, _iter_candidate_edge_masks_streaming_fast, _default_verify_witness as _shared_default_verify_witness
except ImportError:
    from .constraints import get_constraints  # optional
    from .matcher import backchase_repair_cost, find_pattern_matches, MatchResult
    from .grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective
    from .apxchase import _generate_candidate_edge_masks, _iter_candidate_edge_masks_streaming_fast, _default_verify_witness as _shared_default_verify_witness


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


def _parse_edge_spec(edge: Sequence) -> Tuple[Any, Any, Optional[set]]:
    if not isinstance(edge, (tuple, list)):
        raise ValueError(f"Unsupported edge spec: {edge}")
    if len(edge) == 2:
        return edge[0], edge[1], None
    if len(edge) != 3:
        raise ValueError(f"Unsupported edge spec length: {edge}")
    rel_spec = edge[2]
    if rel_spec is None:
        return edge[0], edge[1], None
    if isinstance(rel_spec, dict):
        rel_allowed = rel_spec.get("rel_in", rel_spec.get("in", []))
        return edge[0], edge[1], set(rel_allowed)
    if isinstance(rel_spec, (list, tuple, set)):
        return edge[0], edge[1], set(rel_spec)
    return edge[0], edge[1], {rel_spec}


def _edge_exists_with_rel(graph: Data, u: int, v: int, allowed_rel: Optional[set]) -> bool:
    if getattr(graph, "edge_index", None) is None or graph.edge_index.numel() == 0:
        return False
    edge_index = graph.edge_index
    rel_tensor = getattr(graph, "edge_rel_type", None)
    for col in range(edge_index.size(1)):
        a = int(edge_index[0, col])
        b = int(edge_index[1, col])
        if not ((a == u and b == v) or (a == v and b == u)):
            continue
        if allowed_rel is None:
            return True
        rel_type = None
        if isinstance(rel_tensor, torch.Tensor) and rel_tensor.numel() == edge_index.size(1):
            rel_type = int(rel_tensor[col])
        if rel_type in allowed_rel:
            return True
    return False


def _singleton_rel_id(allowed_rel: Optional[set]) -> Optional[int]:
    if not allowed_rel or len(allowed_rel) != 1:
        return None
    try:
        return int(next(iter(allowed_rel)))
    except Exception:
        return None

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

# --------------------------------- Core class ---------------------------------

class ExhaustChase:
    """
    ExhaustChase: First exhaustively enforce all TGD rules, then generate candidates.
    
    The key difference from ApxChase is the initial exhaustive enforcement phase,
    which ensures no TGD violations remain before candidate generation begins.
    """
    
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
        max_enforce_iterations: int = 100,  # Maximum iterations for exhaustive enforcement
    ):
        self.model = model
        self.Sigma = Sigma
        if self.Sigma is None:
            self.Sigma = []
        self.L = L  # L hop subgraph
        self.k = k  # window size
        self.B = B  # budget for backchase
        self.alpha = alpha  # conc weight
        self.beta = beta  # rpr weight
        self.gamma = gamma  # coverage weight
        self.verify_witness_fn = verify_witness_fn 
        self.conc_fn = conc_fn  # conciseness
        self.rpr_fn = rpr_fn  # repair penalty
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
        self.max_enforce_iterations = max_enforce_iterations
        self._last_run_stats = {}
        
        # If user did not pass a custom gamma_fn, upgrade to a version
        # that also computes repair costs using backchase on a clean graph.
        if gamma_fn is None:
            self.gamma_fn = self._gamma_with_repair
        else:
            self.gamma_fn = gamma_fn

    @staticmethod
    def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def _reference_edge_indices(self, reference_graph: Data, candidate: Data) -> torch.Tensor:
        ref_map: Dict[Tuple[int, int], List[int]] = {}
        ref_ei = reference_graph.edge_index
        for col in range(ref_ei.size(1)):
            u = int(ref_ei[0, col])
            v = int(ref_ei[1, col])
            key = self._canonical_edge(u, v)
            ref_map.setdefault(key, []).append(col)

        candidate_indices: List[int] = []
        cand_ei = candidate.edge_index
        for col in range(cand_ei.size(1)):
            u = int(cand_ei[0, col])
            v = int(cand_ei[1, col])
            candidate_indices.extend(ref_map.get(self._canonical_edge(u, v), []))

        if not candidate_indices:
            return torch.empty((0,), dtype=torch.long, device=reference_graph.edge_index.device)
        return torch.tensor(sorted(set(candidate_indices)), dtype=torch.long, device=reference_graph.edge_index.device)

    def _attach_verification_reference(self, candidate: Data) -> Data:
        reference_graph: Data = getattr(self, '_verify_reference_graph', None) or getattr(self, '_H_observed', None) or getattr(candidate, '_H_full', None)
        if reference_graph is None:
            return candidate
        candidate._H_full = reference_graph
        candidate._edge_idx_in_full = self._reference_edge_indices(reference_graph, candidate)
        return candidate

    def _project_candidate_to_observed(self, candidate: Data) -> Data:
        """
        Exh may enumerate candidates on the cleaned graph, but witness
        verification and final witness metrics must go back to the original
        observed/query graph. Project the cleaned candidate onto observed-local
        edges while preserving the candidate's node set.
        """
        reference_graph: Data = getattr(self, '_verify_reference_graph', None) or getattr(self, '_H_observed', None) or getattr(candidate, '_H_full', None)
        if reference_graph is None:
            return candidate

        observed_edge_keys: Set[Tuple[int, int]] = set()
        ref_ei = reference_graph.edge_index
        for col in range(ref_ei.size(1)):
            observed_edge_keys.add(self._canonical_edge(int(ref_ei[0, col]), int(ref_ei[1, col])))

        keep_cols: List[int] = []
        cand_ei = candidate.edge_index
        nodes_in_observed = getattr(candidate, '_nodes_in_observed', None)
        if nodes_in_observed is None:
            nodes_in_observed = torch.arange(int(candidate.num_nodes), device=cand_ei.device)
        elif isinstance(nodes_in_observed, torch.Tensor):
            nodes_in_observed = nodes_in_observed.clone()
        else:
            nodes_in_observed = torch.tensor(list(nodes_in_observed), dtype=torch.long, device=cand_ei.device)

        for col in range(cand_ei.size(1)):
            u_local = int(cand_ei[0, col])
            v_local = int(cand_ei[1, col])
            if u_local >= len(nodes_in_observed) or v_local >= len(nodes_in_observed):
                continue
            u_obs = int(nodes_in_observed[u_local])
            v_obs = int(nodes_in_observed[v_local])
            if self._canonical_edge(u_obs, v_obs) in observed_edge_keys:
                keep_cols.append(col)

        keep_mask = torch.zeros(cand_ei.size(1), dtype=torch.bool, device=cand_ei.device)
        if keep_cols:
            keep_mask[torch.tensor(keep_cols, dtype=torch.long, device=cand_ei.device)] = True

        projected = Data(
            x=candidate.x.clone() if getattr(candidate, 'x', None) is not None else None,
            edge_index=cand_ei[:, keep_mask].clone(),
        )
        projected.num_nodes = int(candidate.num_nodes)
        if getattr(candidate, 'batch', None) is not None:
            projected.batch = candidate.batch.clone()
        if getattr(candidate, 'y_ref', None) is not None:
            projected.y_ref = candidate.y_ref.clone()
        if getattr(candidate, 'y', None) is not None:
            projected.y = candidate.y.clone()
        if getattr(candidate, 'y_type', None) is not None:
            projected.y_type = candidate.y_type.clone()
        if getattr(candidate, 'node_labels', None) is not None:
            projected.node_labels = candidate.node_labels.clone()
        if getattr(candidate, 'task', None) is not None:
            projected.task = candidate.task
        if getattr(candidate, 'root', None) is not None:
            projected.root = candidate.root
        if getattr(candidate, 'E_base', None) is not None:
            projected.E_base = candidate.E_base
        if getattr(candidate, '_target_node_subgraph_id', None) is not None:
            projected._target_node_subgraph_id = candidate._target_node_subgraph_id
        if getattr(candidate, '_nodes_in_full', None) is not None:
            projected._nodes_in_full = candidate._nodes_in_full.clone()
        projected._nodes_in_observed = nodes_in_observed.clone()
        if getattr(candidate, 'edge_rel_type', None) is not None:
            projected.edge_rel_type = candidate.edge_rel_type[keep_mask].clone()

        projected = self._attach_verification_reference(projected)
        return projected

    def _log(self, msg: str):
        # 输出调试信息
        if self.debug:
            print(f"[ExhaustChase][DEBUG] {msg}")
    # -------------------------------- Main method --------------------------------

    def _gamma_with_repair(self, Gs: Data, Sigma: Sequence, B: int) -> Set[str]:
        if find_pattern_matches is None or Sigma is None or backchase_repair_cost is None:
            return set()
        G_observed: Data = getattr(self, '_H_observed', None) or Gs
        return evaluate_grounding(Gs, Sigma, B, G_observed, find_pattern_matches, backchase_repair_cost)

    def _exhaustive_enforce(self, H: Data) -> Tuple[Data, float, int]:
        """
        Exhaustively enforce all TGDs with standard forward chase semantics.

        For each rule φ = (P, c):
          - match antecedent P on the current observed-local graph
          - if the consequent edge c is missing under that same binding, add it

        This clean/materialization phase is distinct from witness grounding:
        backchase is only used later when evaluating witness coverage.
        
        Returns:
            - Cleaned graph (Data)
            - Time spent on enforcement (float)
            - Number of iterations (int)
        """
        if self.debug:
            print(f"[ExhaustChase] 开始穷尽式规则修复...")
        enforce_start = time.time()
        
        H_clean = H.clone()
        iteration = 0
        total_repairs = 0
        
        while iteration < self.max_enforce_iterations:
            iteration += 1
            violations_found = False
            repairs_this_iter = 0
            
            if self.debug:
                self._log(f"Enforcement iteration {iteration}: checking {len(self.Sigma)} TGDs")
            
            # Check all TGDs for forward-chase violations
            for tgd in self.Sigma:
                try:
                    name = tgd.get('name', 'unnamed') if isinstance(tgd, dict) else str(tgd)
                except Exception:
                    name = str(tgd)
                
                antecedent = tgd.get("antecedent", {}) if isinstance(tgd, dict) else {}
                consequent = tgd.get("consequent", {}) if isinstance(tgd, dict) else {}
                consequent_edges = list(consequent.get("edges", []))
                if not antecedent or not consequent_edges:
                    continue

                # Standard chase trigger: match P on the current graph.
                try:
                    matches = find_pattern_matches(H_clean, antecedent)
                except Exception:
                    matches = []
                
                if not matches:
                    continue
                
                # For each antecedent match, enforce the consequent edge(s).
                for binding in matches:
                    try:
                        for edge_spec in consequent_edges:
                            u_var, v_var, allowed_rel = _parse_edge_spec(edge_spec)
                            if u_var not in binding or v_var not in binding:
                                if self.debug:
                                    self._log(
                                        f"  Skip chase repair for '{name}': consequent vars "
                                        f"({u_var}, {v_var}) are not bound by antecedent match."
                                    )
                                continue

                            u = int(binding[u_var])
                            v = int(binding[v_var])
                            if _edge_exists_with_rel(H_clean, u, v, allowed_rel):
                                continue

                            violations_found = True
                            repairs_this_iter += 1

                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=H_clean.edge_index.device)
                            H_clean.edge_index = torch.cat([H_clean.edge_index, new_edge], dim=1)
                            if hasattr(H_clean, "edge_rel_type") and H_clean.edge_rel_type is not None:
                                rel_id = _singleton_rel_id(allowed_rel)
                                if rel_id is None:
                                    new_rel = torch.full((1,), -1, dtype=H_clean.edge_rel_type.dtype, device=H_clean.edge_rel_type.device)
                                else:
                                    new_rel = torch.full((1,), rel_id, dtype=H_clean.edge_rel_type.dtype, device=H_clean.edge_rel_type.device)
                                H_clean.edge_rel_type = torch.cat([H_clean.edge_rel_type, new_rel], dim=0)
                            if self.debug:
                                self._log(
                                    f"  Chase repaired '{name}': added consequent edge ({u}, {v})"
                                )
                    except Exception as e:
                        if self.debug:
                            self._log(f"  Error checking/repairing TGD '{name}': {e}")
                        continue
            
            total_repairs += repairs_this_iter
            
            # Only print progress periodically to reduce output
            if self.debug or (iteration % 20 == 0 and iteration > 0):
                print(f"[ExhaustChase] 迭代 {iteration}: 修复了 {repairs_this_iter} 个违规, "
                      f"图边数: {H_clean.edge_index.size(1)}")
            
            # If no violations found in this iteration, we're done
            if not violations_found:
                if self.debug or iteration > 1:
                    print(f"[ExhaustChase] 穷尽式修复完成! 迭代次数: {iteration}, 总修复数: {total_repairs}")
                break
        
        if iteration >= self.max_enforce_iterations:
            if self.debug:
                print(f"[ExhaustChase] 警告: 达到最大迭代次数 {self.max_enforce_iterations}, 可能仍有违规存在")
        
        enforce_end = time.time()
        enforce_time = enforce_end - enforce_start
        
        if self.debug:
            print(f"[ExhaustChase] 修复阶段用时: {enforce_time:.4f}秒")
            print(f"[ExhaustChase] 清理后图: |V|={H_clean.num_nodes}, |E|={H_clean.edge_index.size(1)}")
        
        return H_clean, enforce_time, iteration

    # ---------------------------- Public entry points ----------------------------
    def explain_node(self, data: Data, v_t: int) -> Tuple[Set, List[Data], float]:
        """
        Run ExhaustChase for a single target node.
        
        Returns:
            - Sigma*: Set of grounded constraints
            - S_k: List of witness candidates
            - enforce_time: Time spent on exhaustive enforcement
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
        self._verify_reference_graph = data
        original_target = int(v_t)
        try:
            if getattr(H, '_nodes_in_full', None) is not None:
                original_target = int(H._nodes_in_full[int(v_t)].item())
        except Exception:
            original_target = int(v_t)
        
        self._log(f"Start explain_node: v_t={v_t}, |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}")
        print(f"[Exh | Node {original_target}] stage=enforce_constraints")
        
        # Exhaustive enforcement phase
        H_clean, enforce_time, iterations = self._exhaustive_enforce(H)
        print(
            f"[Exh | Node {original_target}] stage=generate_candidates "
            f"|V|={H_clean.num_nodes}, |E|={H_clean.edge_index.size(1)}, "
            f"enforce_iters={iterations}, enforce_time={enforce_time:.4f}s"
        )
        
        # Keep the materialized chased graph for the baseline's own enumeration path.
        self._H_clean = H_clean
        
        # Now run candidate generation on the cleaned graph (same as ApxChase)
        Sigma_star, S_k = self._run(H_clean, root=v_t)
        return Sigma_star, S_k, enforce_time

    def explain_graph(self, data: Data) -> Tuple[Set, List[Data], float]:
        """
        Run ExhaustChase for a graph-level task.
        
        Returns:
            - Sigma*: Set of grounded constraints
            - S_k: List of witness candidates
            - enforce_time: Time spent on exhaustive enforcement
        """
        H = data.clone()
        H.task = 'graph'
        H.root = None
        if getattr(H, 'num_nodes', None) is None and getattr(H, 'x', None) is not None:
            H.num_nodes = H.x.size(0)
        H._nodes_in_full = torch.arange(int(H.num_nodes), device=H.edge_index.device)
        H._nodes_in_observed = torch.arange(int(H.num_nodes), device=H.edge_index.device)
        H.E_base = H.edge_index.size(1)
        self._H_observed = data
        self._verify_reference_graph = data
        
        self._log(f"Start explain_graph: |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}")
        print("[Exh | Graph] stage=enforce_constraints")
        
        # Exhaustive enforcement phase
        H_clean, enforce_time, iterations = self._exhaustive_enforce(H)
        print(
            f"[Exh | Graph] stage=generate_candidates "
            f"|V|={H_clean.num_nodes}, |E|={H_clean.edge_index.size(1)}, "
            f"enforce_iters={iterations}, enforce_time={enforce_time:.4f}s"
        )
        
        # Keep the materialized chased graph for the baseline's own enumeration path.
        self._H_clean = H_clean
        
        # Now run candidate generation on the cleaned graph (same as ApxChase)
        Sigma_star, S_k = self._run(H_clean, root=None)
        
        return Sigma_star, S_k, enforce_time

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
        pair_score = pair_quality(Gs, self.conc_fn, self.alpha, self.beta)
        Gamma_G = self.gamma_fn(H_view, self.Sigma, self.B)
        new_cov = Gamma_G - covered
        if self.debug:
            names_all = _constraint_names(Gamma_G)
            names_new = _constraint_names(new_cov)
            self._log(f"Gamma(G)={len(Gamma_G)} (new={len(new_cov)}); names(new)={names_new[:6]}{'...' if len(names_new)>6 else ''}")
        if self.debug:
            self._log(f"Scores: conc={self.conc_fn(Gs):.4f}, aln={getattr(Gs, '_alignment', 0.0):.4f}, q={pair_score:.4f}")

        current_graphs = [entry[2] for entry in W_k]
        total_constraints = len(self.Sigma)
        current_obj = window_objective(current_graphs, total_constraints, self.gamma)

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
            return window_coverage(trial_graphs)

        if len(Gamma_G) == 0:
            self._log("Skip replacement: no grounded constraints on this candidate.")
            return covered

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
            return covered

        W_k.clear()
        for graph in best_graphs:
            heapq.heappush(W_k, WindowEntry(float(getattr(graph, '_pair_quality', 0.0)), graph).as_tuple())
        return window_coverage(best_graphs)

    def _run(self, H: Data, root: Optional[int]) -> Tuple[Set, List[Data]]:
        # shells of edge indices
        shells = _edge_shells_by_hop(H, root=root, L=self.L)
        self._log(f"Edge shells: {len(shells)} levels; total edges M={H.edge_index.size(1)}")
        # Store full masked/induced graph for reuse in _update_window
        self._H_full = H
        candidate_plan, candidate_generation_mode, large_graph_fastpath = self._candidate_generation_plan(H)
        stream_mode = bool(candidate_plan.pop('stream_mode', False))
        stream_max_candidates = int(candidate_plan.pop('stream_max_candidates', 0) or 0)
        if stream_mode:
            candidate_iter = _iter_candidate_edge_masks_streaming_fast(
                H,
                root=root,
                shells=shells,
                seed_budget=int(candidate_plan.get('seed_per_constraint', self.seed_per_constraint))
                * max(1, int(candidate_plan.get('branch_factor', self.candidate_branch_factor))),
                max_total_candidates=max(1, stream_max_candidates),
            )
            distinct_candidate_count = 0
        else:
            candidate_masks = _generate_candidate_edge_masks(
                H,
                root=root,
                shells=shells,
                Sigma=self.Sigma,
                **candidate_plan,
            )
            candidate_iter = iter(candidate_masks)
            distinct_candidate_count = int(len(candidate_masks))
        W_k: List[Tuple[float, Data]] = []
        covered: Set = set()

        n_candidates = 0
        n_verified = 0
        n_admitted = 0
        fallback_used = False

        for edge_mask in candidate_iter:
            if self.debug:
                self._log(f"Candidate #{n_candidates+1}: |E(G_s)|={int(edge_mask.sum().item())}")
            n_candidates += 1
            if stream_mode:
                distinct_candidate_count += 1
            Gs_clean = _induce_subgraph_from_edges(H, edge_mask)
            Gs = self._project_candidate_to_observed(Gs_clean)
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
            if len(covered) >= len(self.Sigma):
                if self.debug:
                    self._log(f"Early stop: all {len(self.Sigma)} constraints grounded!")
                break
        if len(W_k) == 0:
            H = self._project_candidate_to_observed(H)
            if self.verify_witness_fn(self.model, root, H):
                fallback_used = True
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
        full_witness_edges = extract_witness_edges_in_full(H) if hasattr(H, '_nodes_in_full') else set()
        fallback_selected = any(extract_witness_edges_in_full(Gs) == full_witness_edges for Gs in S_k)
        self._last_run_stats = {
            'num_candidates_generated': int(n_candidates),
            'distinct_candidates_generated': int(distinct_candidate_count),
            'num_candidates_verified': int(n_verified),
            'num_candidates_admitted': int(n_admitted),
            'num_selected_witnesses': int(len(S_k)),
            'num_covered_constraints': int(len(Sigma_star)),
            'fallback_used': bool(fallback_used),
            'fallback_selected': bool(fallback_selected),
            'candidate_generation_mode': str(candidate_generation_mode),
            'large_graph_fastpath': bool(large_graph_fastpath),
            'input_num_nodes': int(getattr(H, 'num_nodes', 0) or 0),
            'input_num_edges': int(H.edge_index.size(1)) if getattr(H, 'edge_index', None) is not None else 0,
        }
        return Sigma_star, S_k

    def _candidate_generation_plan(self, H: Data) -> Tuple[Dict[str, Any], str, bool]:
        num_nodes = int(getattr(H, 'num_nodes', 0) or 0)
        num_edges = int(H.edge_index.size(1)) if getattr(H, 'edge_index', None) is not None else 0
        large_graph_fastpath = self.large_graph_fast_mode and (
            num_nodes >= self.large_graph_node_threshold or num_edges >= self.large_graph_edge_threshold
        )
        kwargs: Dict[str, Any] = {
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
