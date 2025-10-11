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
from typing import Callable, List, Optional, Sequence, Set, Tuple
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

from constraints import get_constraints  # optional
from matcher import Gamma, backchase_repair_cost, find_head_matches, MatchResult


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
        out = model(Gs)
        factual_ok = False
        if hasattr(Gs, 'task') and Gs.task == 'node' and v_t is not None:
            y_hat = out.argmax(dim=-1)
            y_ref = getattr(Gs, 'y_ref', None)
            if y_ref is None:
                factual_ok = True
            else:
                factual_ok = (y_ref[int(v_t)] == y_hat[int(v_t)])
        else:
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
            out_minus = model(H_minus)
            if hasattr(Gs, 'task') and Gs.task == 'node' and v_t is not None:
                y_hat_minus = out_minus.argmax(dim=-1)
                y_hat_gs = out.argmax(dim=-1)
                # Counterfactual: prediction flips for v_t
                counterfactual_ok = (y_hat_gs[int(v_t)] != y_hat_minus[int(v_t)])
            else:
                y_hat_minus = out_minus.argmax(dim=-1)
                y_hat_gs = out.argmax(dim=-1)
                # Counterfactual: prediction flips for graph
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

def _induce_subgraph_from_edges(H: Data, edge_mask: Tensor) -> Data:
    """
    Build a PyG Data subgraph induced by the edges with mask==True.
    Keeps node features for nodes touched by kept edges and (if isolated) the target node H.root.
    Attaches references to the full graph and chosen edge indices for counterfactual verification.
    """
    ei = H.edge_index
    kept_ei = ei[:, edge_mask]
    if kept_ei.numel() > 0:
        nodes = torch.unique(kept_ei.flatten()).to(torch.long)
    else:
        # No edges yet: keep only the root if provided; otherwise keep node 0.
        root_idx = getattr(H, 'root', None)
        if root_idx is None:
            nodes = torch.tensor([0], dtype=torch.long, device=ei.device)
        else:
            nodes = torch.tensor([int(root_idx)], dtype=torch.long, device=ei.device)

    # Build a compact mapping: original node id -> [0..num_nodes-1] in the candidate
    mapping = -torch.ones(int(H.num_nodes if getattr(H, 'num_nodes', None) is not None else H.x.size(0)),
                         dtype=torch.long, device=ei.device)
    mapping[nodes] = torch.arange(nodes.numel(), device=ei.device, dtype=torch.long)

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
    if hasattr(H, 'y_ref'):
        data.y_ref = H.y_ref
    data.root = getattr(H, 'root', None)
    data.E_base = getattr(H, 'E_base', None)
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
        self.debug = debug
        self.max_enforce_iterations = max_enforce_iterations
        
        # If user did not pass a custom gamma_fn, upgrade to a version
        # that also computes repair costs using backchase on a clean graph.
        if gamma_fn is None:
            self.gamma_fn = self._gamma_with_repair
        else:
            self.gamma_fn = gamma_fn

    def _log(self, msg: str):
        # 输出调试信息
        if self.debug:
            print(f"[ExhaustChase][DEBUG] {msg}")
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

    def _exhaustive_enforce(self, H: Data) -> Tuple[Data, float, int]:
        """
        Exhaustively enforce all TGD rules until no violations remain.
        
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
            
            # Check all TGDs for violations
            for tgd in self.Sigma:
                try:
                    name = tgd.get('name', 'unnamed') if isinstance(tgd, dict) else str(tgd)
                except Exception:
                    name = str(tgd)
                
                # Find head matches
                try:
                    matches = find_head_matches(H_clean, tgd)
                except Exception:
                    matches = []
                
                if not matches:
                    continue
                
                # Check each match for body violations and repair if needed
                for binding in matches:
                    try:
                        # Check if body is satisfied
                        feasible, rep_cost, repairs = backchase_repair_cost(H_clean, tgd, binding, self.B)
                        
                        if not feasible or rep_cost > 0:
                            # There is a violation - need to repair
                            violations_found = True
                            repairs_this_iter += 1
                            
                            # Apply repairs by adding missing edges
                            for u, v in repairs:
                                if u >= 0 and v >= 0:  # Skip placeholder repairs (-1, -1)
                                    # Add edge to H_clean
                                    new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=H_clean.edge_index.device)
                                    H_clean.edge_index = torch.cat([H_clean.edge_index, new_edge], dim=1)
                                    if self.debug:
                                        self._log(f"  Repaired: added edge ({u}, {v}) for TGD '{name}'")
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
        H = self._prepare_subgraph(data, v_t)
        H.task = 'node'
        H.root = int(v_t)
        self._H_clean = getattr(data, '_clean', data)
        
        self._log(f"Start explain_node: v_t={v_t}, |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}")
        
        # Exhaustive enforcement phase
        H_clean, enforce_time, iterations = self._exhaustive_enforce(H)
        
        # Update the clean graph reference
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
        H.E_base = H.edge_index.size(1)
        self._H_clean = getattr(data, '_clean', data)
        
        self._log(f"Start explain_graph: |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}")
        
        # Exhaustive enforcement phase
        H_clean, enforce_time, iterations = self._exhaustive_enforce(H)
        
        # Update the clean graph reference
        self._H_clean = H_clean
        
        # Now run candidate generation on the cleaned graph (same as ApxChase)
        Sigma_star, S_k = self._run(H_clean, root=None)
        
        return Sigma_star, S_k, enforce_time

    # ------------------------------ Internal logic ------------------------------
    def _prepare_subgraph(self, data: Data, v_t: int) -> Data:
        """Extract L-hop subgraph around v_t (node task)."""
        node_idx, ei, _, _ = k_hop_subgraph(v_t, self.L, data.edge_index, relabel_nodes=True)
        x = data.x[node_idx] if getattr(data, 'x', None) is not None else None
        out = Data(x=x, edge_index=ei)
        out._nodes_in_full = node_idx.clone()
        out.num_nodes = int(node_idx.numel())
        # carry y_ref if provided (for verify_witness default)
        if hasattr(data, 'y_ref'):
            out.y_ref = data.y_ref
        if hasattr(data, 'batch'):
            out.batch = torch.zeros(out.num_nodes, dtype=torch.long, device=ei.device)
        out.E_base = out.edge_index.size(1)
        return out

    def _update_window(self, W_k: List[Tuple[float, Data]], Gs: Data, covered: Set) -> Set:
        """Update streaming window per paper's UpdateWindow.
        Returns the updated coverage set Γ(W_k).
        """
        # Use the candidate itself for head matching / Γ evaluation
        H_view = Gs
        if self.debug:
            self._log(f"Candidate view: |V|={H_view.num_nodes}, |E|={H_view.edge_index.size(1)}")
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
        new_cov = Gamma_G - covered
        if self.debug:
            names_all = _constraint_names(Gamma_G)
            names_new = _constraint_names(new_cov)
            self._log(f"Gamma(G)={len(Gamma_G)} (new={len(new_cov)}); names(new)={names_new[:6]}{'...' if len(names_new)>6 else ''}")
        if len(Gamma_G) == 0:
            self._log("Skip: no grounded constraints on this candidate.")
            return covered
        if len(new_cov) == 0:
            self._log("Skip: grounded constraints bring no *new* coverage vs window.")
            return covered
        # compute marginal score (conc/rpr on Gs)
        conc = self.conc_fn(Gs)
        rpr = self.rpr_fn(Gs)
        delta = self.alpha * conc + self.beta * rpr + self.gamma * (len(new_cov) / max(1, len(self.Sigma)))
        if self.debug:
            self._log(f"Scores: conc={conc:.4f}, rpr={rpr:.4f}, delta={delta:.4f}")
        entry = WindowEntry(delta, Gs)
        if len(W_k) < self.k:
            heapq.heappush(W_k, entry.as_tuple())
            self._log(f"Heap push (|W_k| -> {len(W_k)}).")
            covered = covered | new_cov
        else:
            if delta > W_k[0][0]:
                self._log(f"Heap replace: popped min delta={W_k[0][0]:.4f}, pushed delta={delta:.4f}.")
                heapq.heapreplace(W_k, entry.as_tuple())
                covered = covered | new_cov
            else:
                self._log(f"Skip: delta={delta:.4f} <= heap-min={W_k[0][0]:.4f}.")
        return covered

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
            # fallback: put H itself if nothing passed verification
            covered = self._update_window(W_k, H, covered)

        final_nodes = (W_k[0][2].num_nodes if len(W_k) > 0 else 0)
        self._log(f"Run stats: candidates={n_candidates}, verified={n_verified}, admitted={n_admitted}, final |W_k|={len(W_k)}, |Γ(W_k)|={len(covered)}, final_nodes={final_nodes}")
        if len(W_k) == 0 and self.debug:
            self._log("No candidates admitted. Consider: increase budget B, relax VerifyWitness, or ensure masking removes head-edges so backchase can trigger.")

        S_k = [entry[2] for entry in sorted(W_k, key=lambda t: -t[0])]
        Sigma_star = covered
        # Annotate each witness with its grounded constraints (names) and repair sum
        annotated = []
        for Gs in S_k:
            # Run Γ on the witness itself (head on Gs; cost vs clean is handled inside gamma_fn)
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