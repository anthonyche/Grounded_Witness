"""
ApxChase: streaming edge-insertion chase for witness generation.

This module implements the pseudocode from the paper in a practical,
pluggable form using PyTorch Geometric. It supports both node-level
and graph-level settings:
  • Node classification (e.g., BAShape): provide a target node id v_t.
  • Graph classification (e.g., MUTAG): leave v_t=None; the whole graph
    is treated as the L-hop region and edges are processed in one shell.

External hooks (pluggable) — functions with the `_fn` suffix can be overridden by users:
  - verify_witness_fn(model, v_t, data_subgraph) -> bool
  - gamma_fn(data_subgraph, Sigma, B) -> Set[Constraint]  (uses matcher.Gamma)
  - conc_fn(data_subgraph) -> float
  - rpr_fn(data_subgraph) -> float
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import heapq

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
    from matcher import backchase_repair_cost, find_pattern_matches, iter_pattern_matches, MatchResult
    from grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective
except ImportError:
    from .constraints import get_constraints  # optional
    from .matcher import backchase_repair_cost, find_pattern_matches, iter_pattern_matches, MatchResult
    from .grounding_semantics import evaluate_grounding, extract_witness_edges_in_full, pair_quality, window_coverage, window_objective


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


@dataclass
class SeedInfo:
    constraint_name: str
    mask: Tensor
    estimated_grounding_cost: int
    consequent_match_count: int
    target_relevance: float
    priority: float


def _edge_mask_signature(edge_mask: Tensor) -> Tuple[int, ...]:
    return tuple(torch.nonzero(edge_mask, as_tuple=False).flatten().detach().cpu().tolist())


def _mask_edge_set(H: Data, edge_mask: Tensor) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for idx in torch.nonzero(edge_mask, as_tuple=False).flatten().tolist():
        u = int(H.edge_index[0, idx])
        v = int(H.edge_index[1, idx])
        out.add((u, v) if u <= v else (v, u))
    return out


def _mask_contains(mask: Tensor, other: Tensor) -> bool:
    return not bool((other & (~mask)).any())


def _node_hops(H: Data, root: Optional[int]) -> Optional[List[int]]:
    if root is None:
        return None
    from collections import deque

    N = int(H.num_nodes if getattr(H, 'num_nodes', None) is not None else H.x.size(0))
    adj = [[] for _ in range(N)]
    und = to_undirected(H.edge_index)
    for u, v in und.t().tolist():
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    dist = [-1] * N
    q = deque([int(root)])
    dist[int(root)] = 0
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def _mask_target_relevance(H: Data, edge_mask: Tensor, root: Optional[int], hops: Optional[List[int]], L: int) -> float:
    if root is None or hops is None:
        return 0.5
    nodes = _nodes_from_mask(H, edge_mask, root)
    if not nodes:
        return 0.0
    finite_hops = [hops[int(node)] for node in nodes if 0 <= int(node) < len(hops) and hops[int(node)] >= 0]
    if not finite_hops:
        return 0.0
    avg_hop = sum(finite_hops) / len(finite_hops)
    return max(0.0, 1.0 - (avg_hop / max(1, L)))


def _constraint_name(tgd: object) -> str:
    if isinstance(tgd, dict):
        return str(tgd.get('name', str(tgd)))
    return str(getattr(tgd, 'name', tgd))


def _diversity_distance(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    if not sig_a and not sig_b:
        return 0.0
    set_a = set(sig_a)
    set_b = set(sig_b)
    denom = len(set_a | set_b)
    if denom == 0:
        return 0.0
    return 1.0 - (len(set_a & set_b) / float(denom))


def _select_diverse_ranked_masks(
    scored_masks: Sequence[Tuple[float, Tensor]],
    max_keep: int,
    diversity_bonus: float,
) -> List[Tensor]:
    pool: List[Tuple[float, Tuple[int, ...], Tensor]] = []
    seen: Set[Tuple[int, ...]] = set()
    for score, mask in scored_masks:
        sig = _edge_mask_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        pool.append((float(score), sig, mask))

    chosen: List[Tensor] = []
    chosen_sigs: List[Tuple[int, ...]] = []
    while pool and len(chosen) < max_keep:
        best_idx = 0
        best_score = None
        for idx, (base_score, sig, _) in enumerate(pool):
            if not chosen_sigs:
                score = base_score
            else:
                min_dist = min(_diversity_distance(sig, prev) for prev in chosen_sigs)
                score = base_score + diversity_bonus * min_dist
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        _, sig, mask = pool.pop(best_idx)
        chosen.append(mask)
        chosen_sigs.append(sig)
    return chosen


def _edge_index_lookup(H: Data) -> Dict[Tuple[int, int], List[int]]:
    edge_lookup: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(H.edge_index.size(1)):
        u = int(H.edge_index[0, idx])
        v = int(H.edge_index[1, idx])
        key = (u, v) if u <= v else (v, u)
        edge_lookup.setdefault(key, []).append(int(idx))
    return edge_lookup


def _nodes_from_mask(H: Data, edge_mask: Tensor, root: Optional[int]) -> Set[int]:
    if edge_mask.numel() == 0 or edge_mask.sum().item() == 0:
        return {int(root)} if root is not None else set()
    kept = H.edge_index[:, edge_mask]
    return {int(v) for v in kept.flatten().tolist()}


def _consequent_seed_masks(
    H: Data,
    Sigma: Sequence,
    max_seed_per_constraint: int,
    max_matches_per_constraint: Optional[int] = None,
) -> List[Tensor]:
    if find_pattern_matches is None or not Sigma:
        return []

    edge_lookup = _edge_index_lookup(H)
    seeds: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()

    for tgd in Sigma:
        try:
            matches = find_pattern_matches(
                H,
                tgd.get("consequent", {}),
                max_results=max_matches_per_constraint,
            )
        except Exception:
            matches = []
        if not matches:
            continue

        consequent_edges = list(tgd.get("consequent", {}).get("edges", [])) if isinstance(tgd, dict) else []
        added = 0
        for binding in matches:
            mask = torch.zeros(H.edge_index.size(1), dtype=torch.bool, device=H.edge_index.device)
            for edge_spec in consequent_edges:
                u_var, v_var = edge_spec[0], edge_spec[1]
                if u_var not in binding or v_var not in binding:
                    continue
                u = int(binding[u_var])
                v = int(binding[v_var])
                key = (u, v) if u <= v else (v, u)
                for idx in edge_lookup.get(key, []):
                    mask[idx] = True
            if mask.sum().item() == 0:
                continue
            sig = _edge_mask_signature(mask)
            if sig in seen:
                continue
            seen.add(sig)
            seeds.append(mask)
            added += 1
            if added >= max_seed_per_constraint:
                break
    return seeds


def _iter_consequent_seed_masks(
    H: Data,
    Sigma: Sequence,
    max_seed_per_constraint: int,
    max_matches_per_constraint: Optional[int] = None,
    max_total_candidates: Optional[int] = None,
):
    if (iter_pattern_matches is None and find_pattern_matches is None) or not Sigma:
        return

    edge_lookup = _edge_index_lookup(H)
    seen_global: Set[Tuple[int, ...]] = set()
    emitted = 0

    for tgd in Sigma:
        consequent = tgd.get("consequent", {}) if isinstance(tgd, dict) else {}
        consequent_edges = list(consequent.get("edges", []))
        added = 0
        try:
            if iter_pattern_matches is not None:
                match_iter = iter_pattern_matches(H, consequent, max_results=max_matches_per_constraint)
            else:
                match_iter = iter(find_pattern_matches(H, consequent, max_results=max_matches_per_constraint))
        except Exception:
            match_iter = iter(())

        for binding in match_iter:
            mask = torch.zeros(H.edge_index.size(1), dtype=torch.bool, device=H.edge_index.device)
            for edge_spec in consequent_edges:
                u_var, v_var = edge_spec[0], edge_spec[1]
                if u_var not in binding or v_var not in binding:
                    continue
                u = int(binding[u_var])
                v = int(binding[v_var])
                key = (u, v) if u <= v else (v, u)
                for idx in edge_lookup.get(key, []):
                    mask[idx] = True
            if mask.sum().item() == 0:
                continue
            sig = _edge_mask_signature(mask)
            if sig in seen_global:
                continue
            seen_global.add(sig)
            yield mask.clone()
            emitted += 1
            added += 1
            if max_total_candidates is not None and emitted >= int(max_total_candidates):
                return
            if added >= max_seed_per_constraint:
                break


def _consequent_seed_infos(
    H: Data,
    Sigma: Sequence,
    max_seed_per_constraint: int,
    pool_factor: int,
    local_budget: int,
    root: Optional[int],
    L: int,
    diversity_bonus: float,
    max_matches_per_constraint: Optional[int] = None,
) -> List[SeedInfo]:
    if find_pattern_matches is None or not Sigma:
        return []

    hops = _node_hops(H, root)
    edge_lookup = _edge_index_lookup(H)
    infos: List[SeedInfo] = []
    scan_limit = max(1, int(max_seed_per_constraint)) * max(1, int(pool_factor))

    for tgd in Sigma:
        try:
            matches = find_pattern_matches(
                H,
                tgd.get("consequent", {}),
                max_results=max_matches_per_constraint,
            )
        except Exception:
            matches = []
        if not matches:
            continue

        consequent_edges = list(tgd.get("consequent", {}).get("edges", [])) if isinstance(tgd, dict) else []
        per_constraint: List[Tuple[float, SeedInfo]] = []
        seen_local: Set[Tuple[int, ...]] = set()
        for binding in matches:
            if len(per_constraint) >= scan_limit:
                break
            mask = torch.zeros(H.edge_index.size(1), dtype=torch.bool, device=H.edge_index.device)
            for edge_spec in consequent_edges:
                u_var, v_var = edge_spec[0], edge_spec[1]
                if u_var not in binding or v_var not in binding:
                    continue
                u = int(binding[u_var])
                v = int(binding[v_var])
                key = (u, v) if u <= v else (v, u)
                for idx in edge_lookup.get(key, []):
                    mask[idx] = True
            if mask.sum().item() == 0:
                continue
            sig = _edge_mask_signature(mask)
            if sig in seen_local:
                continue
            seen_local.add(sig)
            witness_nodes = _nodes_from_mask(H, mask, root)
            witness_edges = _mask_edge_set(H, mask)
            est_cost = local_budget + 1
            if backchase_repair_cost is not None:
                try:
                    ok, rep_cost, _ = backchase_repair_cost(
                        H,
                        tgd.get("antecedent", {}),
                        binding,
                        local_budget,
                        witness_nodes=witness_nodes,
                        witness_edges=witness_edges,
                        return_details=False,
                    )
                    est_cost = int(rep_cost if ok else max(local_budget + 1, rep_cost))
                except Exception:
                    est_cost = local_budget + 1
            target_rel = _mask_target_relevance(H, mask, root, hops, L)
            size_ratio = float(mask.sum().item()) / max(1, int(H.edge_index.size(1)))
            feasible_bonus = 1.0 if est_cost <= local_budget else 0.0
            priority = (
                2.0 * feasible_bonus
                + 1.5 * (1.0 / (1.0 + float(est_cost)))
                + 0.8 * target_rel
                - 1.1 * size_ratio
            )
            per_constraint.append((
                priority,
                SeedInfo(
                    constraint_name=_constraint_name(tgd),
                    mask=mask.clone(),
                    estimated_grounding_cost=int(est_cost),
                    consequent_match_count=1,
                    target_relevance=float(target_rel),
                    priority=float(priority),
                ),
            ))

        if per_constraint:
            chosen_masks = _select_diverse_ranked_masks(
                [(score, info.mask) for score, info in per_constraint],
                max_keep=max(1, int(max_seed_per_constraint)),
                diversity_bonus=diversity_bonus,
            )
            chosen_sigs = {_edge_mask_signature(mask) for mask in chosen_masks}
            for _, info in sorted(per_constraint, key=lambda item: item[0], reverse=True):
                if _edge_mask_signature(info.mask) in chosen_sigs:
                    infos.append(info)
                    chosen_sigs.remove(_edge_mask_signature(info.mask))
                    if not chosen_sigs:
                        break
    return infos


def _legacy_prefix_masks(
    H: Data,
    root: Optional[int],
    shells: Sequence[Tensor],
    max_checkpoints: int,
) -> List[Tensor]:
    if max_checkpoints <= 0:
        return []
    edge_mask = torch.zeros(H.edge_index.size(1), dtype=torch.bool, device=H.edge_index.device)
    current_nodes = torch.tensor([int(root)], dtype=torch.long, device=H.edge_index.device) if root is not None else torch.tensor([], dtype=torch.long, device=H.edge_index.device)
    stream: List[Tensor] = []
    for shell in shells:
        indices = shell if shell.dtype != torch.bool else torch.nonzero(shell, as_tuple=False).flatten()
        for e_idx in indices:
            u, w = H.edge_index[:, e_idx]
            in_u = (current_nodes == int(u)).any()
            in_w = (current_nodes == int(w)).any()
            if (root is None) or (current_nodes.numel() > 0 and (in_u or in_w)):
                edge_mask[e_idx] = True
                stream.append(edge_mask.clone())
                current_nodes = torch.unique(torch.cat([current_nodes, torch.tensor([int(u), int(w)], device=current_nodes.device)]))
    if len(stream) <= max_checkpoints:
        return stream
    selected: List[Tensor] = []
    for i in range(max_checkpoints):
        pos = int(round(i * (len(stream) - 1) / max(1, max_checkpoints - 1)))
        selected.append(stream[pos])
    return selected


def _iter_graph_prefix_seed_masks(
    H: Data,
    root: Optional[int],
    shells: Sequence[Tensor],
    max_seed_per_shell: int,
    max_total_candidates: Optional[int] = None,
    strategy: str = "early",
):
    if max_seed_per_shell <= 0:
        return

    edge_mask = torch.zeros(H.edge_index.size(1), dtype=torch.bool, device=H.edge_index.device)
    current_nodes: Set[int] = {int(root)} if root is not None else set()
    seen: Set[Tuple[int, ...]] = set()
    emitted = 0

    def _shell_indices(shell: Tensor) -> List[int]:
        if shell.dtype == torch.bool:
            return torch.nonzero(shell, as_tuple=False).flatten().detach().cpu().tolist()
        return shell.detach().cpu().tolist()

    def _accepts(u: int, v: int, nodes: Set[int]) -> bool:
        if root is None:
            return True
        if not nodes:
            return True
        return (u in nodes) or (v in nodes)

    for shell in shells:
        indices = _shell_indices(shell)
        if not indices:
            continue

        if strategy == "spaced":
            shell_nodes = set(current_nodes)
            accepted_total = 0
            for e_idx in indices:
                u = int(H.edge_index[0, e_idx])
                v = int(H.edge_index[1, e_idx])
                if not _accepts(u, v, shell_nodes):
                    continue
                accepted_total += 1
                shell_nodes.add(u)
                shell_nodes.add(v)
            if accepted_total == 0:
                continue
            keep = min(int(max_seed_per_shell), accepted_total)
            checkpoints = sorted({
                int(round(i * (accepted_total - 1) / max(1, keep - 1))) + 1
                for i in range(keep)
            })
            checkpoint_set = set(checkpoints)
        else:
            checkpoint_set = set(range(1, int(max_seed_per_shell) + 1))

        accepted_idx = 0
        for e_idx in indices:
            u = int(H.edge_index[0, e_idx])
            v = int(H.edge_index[1, e_idx])
            if not _accepts(u, v, current_nodes):
                continue
            edge_mask[int(e_idx)] = True
            current_nodes.add(u)
            current_nodes.add(v)
            accepted_idx += 1
            if accepted_idx not in checkpoint_set:
                continue
            sig = _edge_mask_signature(edge_mask)
            if sig in seen:
                continue
            seen.add(sig)
            yield edge_mask.clone()
            emitted += 1
            if max_total_candidates is not None and emitted >= int(max_total_candidates):
                return


def _graph_seed_masks(
    H: Data,
    root: Optional[int],
    shells: Sequence[Tensor],
    seed_budget: int,
    max_total_candidates: Optional[int] = None,
    strategy: str = "spaced",
) -> List[Tensor]:
    if strategy != "diverse":
        return list(
            _iter_graph_prefix_seed_masks(
                H,
                root=root,
                shells=shells,
                max_seed_per_shell=max(1, int(seed_budget)),
                max_total_candidates=max_total_candidates,
                strategy=strategy,
            )
        )

    out: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()

    def _append(mask_iter: Iterable[Tensor]) -> None:
        nonlocal out
        for mask in mask_iter:
            sig = _edge_mask_signature(mask)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(mask.clone())
            if max_total_candidates is not None and len(out) >= int(max_total_candidates):
                return

    def _reverse_shell(shell: Tensor) -> Tensor:
        if shell.dtype == torch.bool:
            return shell.clone()
        return torch.flip(shell, dims=[0])

    reversed_shells = [_reverse_shell(shell) for shell in shells]

    _append(
        _iter_graph_prefix_seed_masks(
            H,
            root=root,
            shells=shells,
            max_seed_per_shell=max(1, int(seed_budget)),
            max_total_candidates=max_total_candidates,
            strategy="early",
        )
    )
    if max_total_candidates is None or len(out) < int(max_total_candidates):
        _append(
            _iter_graph_prefix_seed_masks(
                H,
                root=root,
                shells=shells,
                max_seed_per_shell=max(1, int(seed_budget)),
                max_total_candidates=None if max_total_candidates is None else int(max_total_candidates) - len(out),
                strategy="spaced",
            )
        )
    if max_total_candidates is None or len(out) < int(max_total_candidates):
        _append(
            _iter_graph_prefix_seed_masks(
                H,
                root=root,
                shells=reversed_shells,
                max_seed_per_shell=max(1, int(seed_budget)),
                max_total_candidates=None if max_total_candidates is None else int(max_total_candidates) - len(out),
                strategy="spaced",
            )
        )
    return out


def _bounded_expand_masks(
    H: Data,
    seeds: Sequence[Tensor],
    root: Optional[int],
    expand_steps: int,
    branch_factor: int,
    beam_width: int,
    max_masks: int,
    candidate_score_fn: Optional[Callable[[Tensor], float]] = None,
    diversity_bonus: float = 0.0,
) -> List[Tensor]:
    if not seeds:
        return []

    edge_lookup = _edge_index_lookup(H)
    del edge_lookup  # lookup built for symmetry with seed generation; not needed below
    all_masks: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()
    beam: List[Tensor] = []

    for mask in seeds:
        sig = _edge_mask_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        beam.append(mask.clone())
        all_masks.append(mask.clone())
        if len(all_masks) >= max_masks:
            return all_masks

    for _ in range(max(0, expand_steps)):
        next_candidates: List[Tuple[float, Tensor]] = []
        for mask in beam:
            current_nodes = _nodes_from_mask(H, mask, root)
            frontier: List[Tuple[Tuple[int, int, int], int]] = []
            for e_idx in range(H.edge_index.size(1)):
                if bool(mask[e_idx]):
                    continue
                u = int(H.edge_index[0, e_idx])
                v = int(H.edge_index[1, e_idx])
                if root is not None and current_nodes and (u not in current_nodes and v not in current_nodes):
                    continue
                touches = int(u in current_nodes) + int(v in current_nodes)
                introduces = int(u not in current_nodes) + int(v not in current_nodes)
                score = (touches, introduces, -e_idx)
                frontier.append((score, e_idx))
            frontier.sort(reverse=True)
            shortlist = frontier[:max(branch_factor, beam_width) * (2 if candidate_score_fn is not None else 1)]
            for heuristic_key, e_idx in shortlist:
                child = mask.clone()
                child[e_idx] = True
                sig = _edge_mask_signature(child)
                if sig in seen:
                    continue
                seen.add(sig)
                child_score = (
                    float(candidate_score_fn(child))
                    if candidate_score_fn is not None
                    else float((heuristic_key[0] * 10) + heuristic_key[1] - int(child.sum().item()) * 0.01)
                )
                next_candidates.append((child_score, child))
                all_masks.append(child.clone())
                if len(all_masks) >= max_masks:
                    return all_masks
        if candidate_score_fn is not None:
            beam = _select_diverse_ranked_masks(next_candidates, max_keep=beam_width, diversity_bonus=diversity_bonus)
        else:
            next_candidates.sort(key=lambda item: item[0], reverse=True)
            beam = [mask for _, mask in next_candidates[:beam_width]]
        if not beam:
            break
    return all_masks


def _bounded_prune_full_masks(
    H: Data,
    seed_masks: Sequence[Tensor],
    max_edge_choices: int = 8,
    max_remove_budget: int = 2,
    max_masks: int = 12,
) -> List[Tensor]:
    M = H.edge_index.size(1)
    if M == 0 or max_masks <= 0:
        return []

    edge_scores = [0] * M
    for mask in seed_masks:
        active = torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()
        for idx in active:
            edge_scores[int(idx)] += 1

    ranked = list(range(M))
    ranked.sort(key=lambda idx: (edge_scores[idx], idx))
    candidates = ranked[:max(1, min(max_edge_choices, M))]

    full_mask = torch.ones(M, dtype=torch.bool, device=H.edge_index.device)
    masks: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()

    def _append(mask: Tensor) -> None:
        sig = _edge_mask_signature(mask)
        if sig in seen:
            return
        seen.add(sig)
        masks.append(mask)

    for idx in candidates:
        child = full_mask.clone()
        child[idx] = False
        _append(child)
        if len(masks) >= max_masks:
            return masks

    if max_remove_budget >= 2:
        pair_limit = min(len(candidates), 6)
        for i in range(pair_limit):
            for j in range(i + 1, pair_limit):
                child = full_mask.clone()
                child[candidates[i]] = False
                child[candidates[j]] = False
                _append(child)
                if len(masks) >= max_masks:
                    return masks
    return masks


def _edge_support_from_seed_masks(seed_masks: Sequence[Tensor], num_edges: int) -> List[float]:
    support = [0.0] * int(num_edges)
    for mask in seed_masks:
        for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            support[int(idx)] += 1.0
    return support


def _hybrid_prune_full_masks(
    H: Data,
    seed_masks: Sequence[Tensor],
    candidate_score_fn: Callable[[Tensor], float],
    max_masks: int,
    delete_budget: int,
    branch_factor: int,
    beam_width: int,
    root: Optional[int],
    diversity_bonus: float,
) -> List[Tensor]:
    M = int(H.edge_index.size(1))
    if M == 0 or max_masks <= 0 or delete_budget <= 0:
        return []

    graph_mode = root is None
    edge_support = _edge_support_from_seed_masks(seed_masks, M)
    hops = _node_hops(H, root)
    full_mask = torch.ones(M, dtype=torch.bool, device=H.edge_index.device)
    results: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()
    beam: List[Tensor] = [full_mask]

    def _delete_priority(mask: Tensor, e_idx: int) -> float:
        u = int(H.edge_index[0, e_idx])
        v = int(H.edge_index[1, e_idx])
        support_penalty = edge_support[e_idx]
        endpoint_hop = 0.0
        if hops is not None:
            h_u = hops[u] if 0 <= u < len(hops) and hops[u] >= 0 else max(1, len(hops))
            h_v = hops[v] if 0 <= v < len(hops) and hops[v] >= 0 else max(1, len(hops))
            endpoint_hop = float(h_u + h_v) / 2.0
        # Higher means "better edge to delete".
        if graph_mode:
            return (1.5 * (1.0 / (1.0 + support_penalty))) + 0.05 * endpoint_hop
        return (1.0 * (1.0 / (1.0 + support_penalty))) + 0.20 * endpoint_hop

    for step in range(max(1, int(delete_budget))):
        next_pool: List[Tuple[float, Tensor]] = []
        for mask in beam:
            active_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            ranked_edges = sorted(active_indices, key=lambda idx: (_delete_priority(mask, int(idx)), -int(idx)), reverse=True)
            for e_idx in ranked_edges[:max(1, int(branch_factor))]:
                child = mask.clone()
                child[int(e_idx)] = False
                sig = _edge_mask_signature(child)
                if sig in seen:
                    continue
                seen.add(sig)
                delete_gain = (M - int(child.sum().item())) / float(max(1, M))
                score = float(candidate_score_fn(child))
                # Encourage concise near-full candidates without collapsing into tiny graphs.
                score += (1.2 if graph_mode else 0.35) * delete_gain
                next_pool.append((score, child))
                results.append(child.clone())
                if len(results) >= max_masks:
                    return results
        if not next_pool:
            break
        beam = _select_diverse_ranked_masks(next_pool, max_keep=max(1, int(beam_width)), diversity_bonus=diversity_bonus)
    return results


def _generate_candidate_edge_masks(
    H: Data,
    root: Optional[int],
    shells: Sequence[Tensor],
    Sigma: Sequence,
    seed_per_constraint: int = 2,
    expand_steps: int = 2,
    branch_factor: int = 3,
    beam_width: int = 6,
    max_candidates: int = 48,
    legacy_checkpoints: int = 6,
    local_budget: int = 2,
    use_ranked_candidate_prioritization: bool = False,
    use_task_aware_hybrid_generation: bool = False,
    ranking_pool_factor: int = 3,
    diversity_bonus: float = 0.20,
    max_near_full_candidates: int = 16,
    near_full_delete_budget: int = 3,
    near_full_branch_factor: int = 6,
    near_full_beam_width: int = 4,
    max_consequent_matches_per_constraint: Optional[int] = None,
    use_full_mask_prune: bool = True,
    use_legacy_prefix_masks: bool = True,
) -> List[Tensor]:
    del Sigma  # Candidate generation is graph-only; constraints are used later in Γ/grounding.
    seed_budget = max(1, int(seed_per_constraint)) * max(1, int(branch_factor))

    if not use_ranked_candidate_prioritization:
        max_candidates = max(1, int(max_candidates))
        candidate_masks: List[Tensor] = []
        seen: Set[Tuple[int, ...]] = set()

        def _append(mask_iter: Iterable[Tensor]) -> None:
            nonlocal candidate_masks
            for mask in mask_iter:
                sig = _edge_mask_signature(mask)
                if sig in seen:
                    continue
                seen.add(sig)
                candidate_masks.append(mask.clone())
                if len(candidate_masks) >= max_candidates:
                    return

        seed_masks = _graph_seed_masks(
            H,
            root=root,
            shells=shells,
            seed_budget=seed_budget,
            max_total_candidates=max_candidates,
            strategy="diverse",
        )
        _append(seed_masks)
        if len(candidate_masks) < max_candidates:
            expanded = _bounded_expand_masks(
                H,
                seed_masks,
                root=root,
                expand_steps=max(0, int(expand_steps)),
                branch_factor=max(1, int(branch_factor)),
                beam_width=max(1, int(beam_width)),
                max_masks=max_candidates - len(candidate_masks),
            )
            _append(expanded)
        if use_full_mask_prune and len(candidate_masks) < max_candidates:
            _append(
                _bounded_prune_full_masks(
                    H,
                    seed_masks,
                    max_edge_choices=min(8, H.edge_index.size(1)),
                    max_remove_budget=2,
                    max_masks=max_candidates - len(candidate_masks),
                )
            )
        if use_legacy_prefix_masks and len(candidate_masks) < max_candidates:
            _append(_legacy_prefix_masks(H, root, shells, max_checkpoints=max(0, int(legacy_checkpoints))))

        if not candidate_masks:
            candidate_masks = _legacy_prefix_masks(H, root, shells, max_checkpoints=1)
        return candidate_masks[:max_candidates]

    max_candidates = max(1, int(max_candidates))
    pool_limit = max_candidates * max(2, int(ranking_pool_factor))
    candidate_masks: List[Tensor] = []
    seen: Set[Tuple[int, ...]] = set()
    hop_cache = _node_hops(H, root)
    score_cache: Dict[Tuple[int, ...], float] = {}

    def _candidate_priority(mask: Tensor) -> float:
        sig = _edge_mask_signature(mask)
        cached = score_cache.get(sig)
        if cached is not None:
            return cached
        size_ratio = float(mask.sum().item()) / max(1, int(H.edge_index.size(1)))
        node_ratio = float(len(_nodes_from_mask(H, mask, root))) / max(1, int(H.num_nodes))
        target_rel = _mask_target_relevance(H, mask, root, hop_cache, max(1, len(shells)))
        edge_count = float(mask.sum().item())
        connectivity_bonus = 1.0 if edge_count > 0 else 0.0
        root_bonus = 1.0 if (root is None or int(root) in _nodes_from_mask(H, mask, root)) else 0.0
        priority = (
            1.6 * target_rel
            + 0.7 * connectivity_bonus
            + 0.5 * root_bonus
            + 0.25 * min(edge_count / max(1.0, float(max(1, len(shells)))), 1.0)
            + 0.8 * target_rel
            - 1.7 * size_ratio
            - 0.5 * node_ratio
        )
        score_cache[sig] = float(priority)
        return float(priority)

    def _append(mask_iter: Iterable[Tensor], limit: int) -> None:
        nonlocal candidate_masks
        for mask in mask_iter:
            sig = _edge_mask_signature(mask)
            if sig in seen:
                continue
            seen.add(sig)
            candidate_masks.append(mask.clone())
            if len(candidate_masks) >= limit:
                return

    seed_masks = _graph_seed_masks(
        H,
        root=root,
        shells=shells,
        seed_budget=seed_budget,
        max_total_candidates=pool_limit,
        strategy="diverse",
    )
    graph_mode = root is None

    if not use_task_aware_hybrid_generation:
        _append(seed_masks, pool_limit)
        if len(candidate_masks) < pool_limit:
            expanded = _bounded_expand_masks(
                H,
                seed_masks,
                root=root,
                expand_steps=max(0, int(expand_steps)),
                branch_factor=max(1, int(branch_factor)),
                beam_width=max(1, int(beam_width)),
                max_masks=pool_limit - len(candidate_masks),
                candidate_score_fn=_candidate_priority,
                diversity_bonus=float(diversity_bonus),
            )
            _append(expanded, pool_limit)
        if use_full_mask_prune and len(candidate_masks) < pool_limit:
            _append(
                _bounded_prune_full_masks(
                    H,
                    seed_masks,
                    max_edge_choices=min(8, H.edge_index.size(1)),
                    max_remove_budget=2,
                    max_masks=pool_limit - len(candidate_masks),
                ),
                pool_limit,
            )
        if use_legacy_prefix_masks and len(candidate_masks) < pool_limit:
            _append(_legacy_prefix_masks(H, root, shells, max_checkpoints=max(0, int(legacy_checkpoints))), pool_limit)
    else:
        if graph_mode:
            legacy_quota = min(max(2, int(legacy_checkpoints)), max(4, max_candidates // 8))
            prune_quota = min(
                int(max_near_full_candidates),
                max(4, max_candidates // 2),
            )
        else:
            legacy_quota = min(max(1, int(legacy_checkpoints)), 2)
            prune_quota = min(int(max_near_full_candidates), 2)
        growth_quota = max(1, max_candidates - prune_quota - legacy_quota)

        growth_masks: List[Tensor] = []
        growth_seen: Set[Tuple[int, ...]] = set()
        for mask in seed_masks:
            sig = _edge_mask_signature(mask)
            if sig not in growth_seen:
                growth_seen.add(sig)
                growth_masks.append(mask.clone())
        expanded = _bounded_expand_masks(
            H,
            seed_masks,
            root=root,
            expand_steps=max(0, int(expand_steps)),
            branch_factor=max(1, int(branch_factor)),
            beam_width=max(1, int(beam_width)),
            max_masks=max(1, growth_quota * max(2, int(ranking_pool_factor))),
            candidate_score_fn=_candidate_priority,
            diversity_bonus=float(diversity_bonus),
        )
        for mask in expanded:
            sig = _edge_mask_signature(mask)
            if sig not in growth_seen:
                growth_seen.add(sig)
                growth_masks.append(mask.clone())

        prune_masks = _hybrid_prune_full_masks(
            H,
            seed_masks=seed_masks,
            candidate_score_fn=_candidate_priority,
            max_masks=max(1, prune_quota * max(2, int(ranking_pool_factor))),
            delete_budget=max(1, int(near_full_delete_budget)) if graph_mode else max(1, min(1, int(near_full_delete_budget))),
            branch_factor=max(1, int(near_full_branch_factor)) if graph_mode else max(1, min(3, int(near_full_branch_factor))),
            beam_width=max(1, int(near_full_beam_width)) if graph_mode else max(1, min(2, int(near_full_beam_width))),
            root=root,
            diversity_bonus=float(diversity_bonus),
        )
        legacy_masks = _legacy_prefix_masks(H, root, shells, max_checkpoints=max(0, int(legacy_checkpoints))) if use_legacy_prefix_masks else []

        branch_selected: List[Tensor] = []
        branch_selected.extend(
            _select_diverse_ranked_masks(
                [(_candidate_priority(mask), mask) for mask in growth_masks],
                max_keep=max(1, growth_quota),
                diversity_bonus=float(diversity_bonus),
            )
        )
        branch_selected.extend(
            _select_diverse_ranked_masks(
                [(_candidate_priority(mask), mask) for mask in prune_masks],
                max_keep=max(1, prune_quota),
                diversity_bonus=float(diversity_bonus),
            )
        )
        branch_selected.extend(
            _select_diverse_ranked_masks(
                [(_candidate_priority(mask), mask) for mask in legacy_masks],
                max_keep=max(0, legacy_quota),
                diversity_bonus=float(diversity_bonus) * 0.5,
            )
        )
        _append(branch_selected, max_candidates)
        if len(candidate_masks) < max_candidates:
            fallback_pool: List[Tensor] = []
            for source in (growth_masks, prune_masks, legacy_masks):
                for mask in source:
                    fallback_pool.append(mask)
            _append(
                _select_diverse_ranked_masks(
                    [(_candidate_priority(mask), mask) for mask in fallback_pool],
                    max_keep=max_candidates - len(candidate_masks),
                    diversity_bonus=float(diversity_bonus),
                ),
                max_candidates,
            )

    if not candidate_masks:
        candidate_masks = _legacy_prefix_masks(H, root, shells, max_checkpoints=1)

    ranked = _select_diverse_ranked_masks(
        [(_candidate_priority(mask), mask) for mask in candidate_masks],
        max_keep=max_candidates,
        diversity_bonus=float(diversity_bonus),
    )
    return ranked[:max_candidates]


def _iter_candidate_edge_masks_streaming_fast(
    H: Data,
    root: Optional[int],
    shells: Sequence[Tensor],
    seed_budget: int,
    max_total_candidates: int,
):
    for mask in _iter_graph_prefix_seed_masks(
        H,
        root=root,
        shells=shells,
        max_seed_per_shell=max(1, int(seed_budget)),
        max_total_candidates=max(1, int(max_total_candidates)),
        strategy="early",
    ):
        yield mask

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
    """Backward-compatible alias for alignment-style grounding quality."""
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
    # FIX: For node task, we should ALWAYS index y/y_ref/y_type by selected nodes
    # Task type should be inherited from H (the full L-hop graph), not inferred
    is_node_task = (task_type == 'node') or \
                   (root_val is not None and root_val >= 0) or \
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
    # Only node-level candidates should expose _target_node_subgraph_id.
    if is_node_task:
        root_val = getattr(data, 'root', None)
        if root_val is not None and root_val >= 0:
            root_idx = int(root_val)
            if root_idx < len(mapping) and mapping[root_idx] >= 0:
                data._target_node_subgraph_id = int(mapping[root_idx].item())
            else:
                data._target_node_subgraph_id = 0
        elif hasattr(H, '_target_node_subgraph_id'):
            old_id = H._target_node_subgraph_id
            if old_id < len(mapping) and mapping[old_id] >= 0:
                data._target_node_subgraph_id = int(mapping[old_id].item())
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
        self._H_observed = data
        # DEBUG: Check H.y distribution
        if hasattr(H, 'y') and H.y is not None:
            y_counts = {}
            for lbl in H.y.tolist():
                y_counts[lbl] = y_counts.get(lbl, 0) + 1
            print(f"[explain_node DEBUG] H.y distribution: {y_counts}")
        else:
            print(f"[explain_node DEBUG] H has no y attribute!")
        self._log(f"Start explain_node: v_t={v_t}, |V(H)|={H.num_nodes}, |E(H)|={H.edge_index.size(1)}, L={self.L}, k={self.k}, B={self.B}, |Sigma|={len(self.Sigma)}")
        if self.debug:
            self._log("Debugging mode — consequent-only diagnostics may be skipped.")
            # Print actual constraint names and their consequent edge counts
            constraint_info = []
            for c in self.Sigma:
                name = c.get('name', 'unnamed')
                consequent_edges = len(c.get("consequent", {}).get("edges", []))
                constraint_info.append(f"{name}({consequent_edges}e)")
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
        # Grounding metadata is defined over full-graph node ids. Graph-level
        # tasks therefore need an explicit identity mapping; otherwise the
        # fallback full-graph witness gets treated as ungroundable.
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
        # carry y (true labels) for matcher
        if hasattr(data, 'y'):
            out.y = data.y[node_idx]  # Extract true labels for subgraph nodes
        # carry y_type (KMeans cluster labels) for TGD matching
        if hasattr(data, 'y_type'):
            out.y_type = data.y_type[node_idx]  # Extract type labels for subgraph nodes
        if hasattr(data, 'node_labels'):
            out.node_labels = data.node_labels[node_idx]
        if hasattr(data, 'edge_rel_type'):
            out.edge_rel_type = data.edge_rel_type[edge_mask]
        if hasattr(data, 'batch'):
            out.batch = torch.zeros(out.num_nodes, dtype=torch.long, device=ei.device)
        out.E_base = out.edge_index.size(1)
        out.root = v_t  # Store original target node ID (in full graph)
        out.task = 'node'
        return out

    def _update_window(self, W_k: List[Tuple[float, Data]], Gs: Data, covered: Set) -> Set:
        """Set-level UpdateWindow approximation for F(W)."""
        H_view = Gs
        if self.debug:
            self._log(f"Candidate view: |V|={H_view.num_nodes}, |E|={H_view.edge_index.size(1)}")
            # DEBUG: Check node label distribution in candidate
            if hasattr(H_view, 'y') and H_view.y is not None:
                labels_in_view = H_view.y.tolist() if H_view.y.numel() <= 20 else H_view.y[:20].tolist()
                label_counts = {}
                for lbl in H_view.y.tolist():
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                self._log(f"Candidate labels: {label_counts} (first 20: {labels_in_view})")
            else:
                self._log(f"WARNING: Candidate has no y attribute!")
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

        # Witness semantics: any verified candidate is already a valid witness.
        # When the window is not full yet, admit the verified witness first and
        # only then let later candidates compete on the set objective.
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
            # Only allow the full observed graph as a fallback witness if it
            # itself satisfies the witness definition.
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
