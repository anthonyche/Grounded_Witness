from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _parse_edge_spec(edge: Any) -> Tuple[Any, Any, Any]:
    if not isinstance(edge, (tuple, list)):
        raise ValueError(f"Unsupported edge spec: {edge}")
    if len(edge) == 2:
        return edge[0], edge[1], None
    if len(edge) == 3:
        return edge[0], edge[1], edge[2]
    raise ValueError(f"Unsupported edge spec length: {edge}")


def _edge_key(edge: Any) -> Tuple[str, str, Any]:
    u, v, rel = _parse_edge_spec(edge)
    a, b = str(u), str(v)
    return (a, b, rel) if a <= b else (b, a, rel)


def _edge_spec_from_key(edge_key: Tuple[str, str, Any]) -> Tuple[Any, ...]:
    u, v, rel = edge_key
    return (u, v) if rel is None else (u, v, rel)


def _pattern_from_edges(nodes_spec: Dict[str, Any], edges: Sequence[Any]) -> Dict[str, Any]:
    used_nodes: Set[str] = set()
    edge_specs: List[Tuple[Any, ...]] = []
    for edge in edges:
        u, v, rel = _parse_edge_spec(edge)
        used_nodes.add(str(u))
        used_nodes.add(str(v))
        edge_specs.append((u, v) if rel is None else (u, v, rel))
    return {
        "nodes": {name: dict(nodes_spec[name]) for name in used_nodes},
        "edges": edge_specs,
        "distinct": sorted(used_nodes),
    }


def interpret_constraint(tgd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a constraint into the standard backchase view:
      antecedent = P
      consequent = c
    """
    antecedent = dict(tgd.get("antecedent", {}))
    consequent = dict(tgd.get("consequent", {}))
    return {
        "antecedent": antecedent,
        "consequent": consequent,
    }


def _workload_graph_nx(workload_graph: Any):
    try:
        from matcher import _data_to_nx
    except ImportError:
        from .matcher import _data_to_nx  # type: ignore
    return _data_to_nx(workload_graph)


def _max_node_assignment(
    pattern: Dict[str, Any],
    binding: Dict[str, int],
    graph_nx: Any,
) -> Tuple[int, Dict[str, int]]:
    node_specs = pattern.get("nodes", {})
    distinct_vars = set(pattern.get("distinct", []))
    labels = {int(n): int(data["label"]) for n, data in graph_nx.nodes(data=True)}
    all_nodes = sorted(labels.keys())
    vars_all = sorted(node_specs.keys())

    for var, node_id in binding.items():
        if var not in node_specs:
            return 0, {}
        allowed = set(node_specs[var].get("in", []))
        if allowed and labels.get(int(node_id)) not in allowed:
            return 0, {}

    best_count = len(binding)
    best_env = dict(binding)
    unbound = [var for var in vars_all if var not in binding]
    candidate_map: Dict[str, List[int]] = {}
    for var in unbound:
        allowed = set(node_specs[var].get("in", []))
        candidate_map[var] = [n for n in all_nodes if not allowed or labels[n] in allowed]

    order = sorted(unbound, key=lambda v: len(candidate_map[v]))

    def dfs(idx: int, env: Dict[str, int]) -> None:
        nonlocal best_count, best_env
        if len(env) > best_count:
            best_count = len(env)
            best_env = dict(env)
        if idx >= len(order):
            return
        remaining = len(order) - idx
        if len(env) + remaining <= best_count:
            return

        var = order[idx]
        blocked = {env[dv] for dv in distinct_vars if dv in env and dv != var} if var in distinct_vars else set()
        for node_id in candidate_map[var]:
            if node_id in blocked:
                continue
            env[var] = node_id
            dfs(idx + 1, env)
            env.pop(var, None)
        dfs(idx + 1, env)

    dfs(0, dict(binding))
    return best_count, best_env


def _consequent_matches(
    workload_graph: Any,
    consequent_pattern: Dict[str, Any],
    find_pattern_matches_fn,
) -> List[Dict[str, int]]:
    if not consequent_pattern.get("edges", []):
        return []
    try:
        return list(find_pattern_matches_fn(workload_graph, consequent_pattern))
    except Exception:
        return []


def constraint_activation_summary(
    workload_graph: Any,
    Sigma: Sequence,
    observed_graph: Any,
    find_pattern_matches_fn,
) -> Dict[str, Any]:
    hit_names: Set[str] = set()
    active_names: Set[str] = set()
    details: List[Dict[str, Any]] = []
    graph_nx = _workload_graph_nx(workload_graph)

    for tgd in Sigma:
        try:
            name = tgd.get("name", "unnamed") if isinstance(tgd, dict) else str(tgd)
        except Exception:
            name = str(tgd)
        try:
            semantics = interpret_constraint(tgd)
            matches = _consequent_matches(workload_graph, semantics["consequent"], find_pattern_matches_fn)
            matched_edge_count = len(semantics["consequent"].get("edges", [])) if matches else 0
        except Exception:
            matches = []
            matched_edge_count = 0
            semantics = interpret_constraint(tgd)

        if not matches:
            continue

        hit_names.add(name)
        antecedent_pattern = semantics["antecedent"]
        antecedent_var_count = len(antecedent_pattern.get("nodes", {}))
        best_assigned_count = 0
        node_complete = False
        for bind_view in matches:
            bind_for_antecedent = {var: node_id for var, node_id in bind_view.items() if var in antecedent_pattern.get("nodes", {})}
            assigned_count, _ = _max_node_assignment(antecedent_pattern, bind_for_antecedent, graph_nx)
            best_assigned_count = max(best_assigned_count, int(assigned_count))
            if int(assigned_count) == antecedent_var_count:
                node_complete = True
                break
        if node_complete:
            active_names.add(name)
        details.append(
            {
                "constraint_name": name,
                "matched_edge_count": int(matched_edge_count),
                "match_count": int(len(matches)),
                "target_var_count": int(antecedent_var_count),
                "best_assigned_count": int(best_assigned_count),
                "node_complete": bool(node_complete),
            }
        )

    return {
        "hit_names": hit_names,
        "active_names": active_names,
        "details": details,
    }


def hit_constraint_names(
    workload_graph: Any,
    Sigma: Sequence,
    observed_graph: Any,
    find_pattern_matches_fn,
) -> Set[str]:
    return set(constraint_activation_summary(workload_graph, Sigma, observed_graph, find_pattern_matches_fn)["hit_names"])


def active_constraint_names(
    workload_graph: Any,
    Sigma: Sequence,
    observed_graph: Any,
    find_pattern_matches_fn,
) -> Set[str]:
    return set(constraint_activation_summary(workload_graph, Sigma, observed_graph, find_pattern_matches_fn)["active_names"])


def extract_witness_edges_in_full(Gs: Any) -> Set[Tuple[int, int]]:
    edge_index = getattr(Gs, 'edge_index', None)
    nodes_in_full = getattr(Gs, '_nodes_in_full', None)
    if edge_index is None or nodes_in_full is None:
        return set()

    if isinstance(nodes_in_full, torch.Tensor):
        node_map = nodes_in_full.detach().cpu().tolist()
    else:
        node_map = list(nodes_in_full)

    edges: Set[Tuple[int, int]] = set()
    for col in range(edge_index.size(1)):
        u_view = int(edge_index[0, col])
        v_view = int(edge_index[1, col])
        if u_view >= len(node_map) or v_view >= len(node_map):
            continue
        edges.add(canonical_edge(int(node_map[u_view]), int(node_map[v_view])))
    return edges


def extract_witness_edges_in_observed(Gs: Any) -> Set[Tuple[int, int]]:
    edge_index = getattr(Gs, 'edge_index', None)
    nodes_in_observed = getattr(Gs, '_nodes_in_observed', None)
    if edge_index is None:
        return set()
    if nodes_in_observed is None:
        nodes_in_observed = list(range(int(Gs.num_nodes)))
    elif isinstance(nodes_in_observed, torch.Tensor):
        nodes_in_observed = nodes_in_observed.detach().cpu().tolist()
    else:
        nodes_in_observed = list(nodes_in_observed)

    edges: Set[Tuple[int, int]] = set()
    for col in range(edge_index.size(1)):
        u_view = int(edge_index[0, col])
        v_view = int(edge_index[1, col])
        if u_view >= len(nodes_in_observed) or v_view >= len(nodes_in_observed):
            continue
        edges.add(canonical_edge(int(nodes_in_observed[u_view]), int(nodes_in_observed[v_view])))
    return edges


def attach_grounding_metadata(
    Gs: Any,
    grounded_names: Iterable[str],
    hit_names: Iterable[str],
    active_names: Iterable[str],
    grounded_details: List[Dict[str, Any]],
    delta_edges: Set[Tuple[int, int]],
    supporting_edges: Set[Tuple[int, int]],
    local_budget: int,
) -> Set[str]:
    # This metadata is an aggregated grounding summary for the witness-level
    # pair view. Each constraint is still checked independently from the same
    # witness G_s; we do not chain repairs across constraints.
    grounded_set = set(grounded_names)
    hit_set = set(hit_names)
    active_set = set(active_names)
    if grounded_set:
        alignment = 1.0 - (len(delta_edges) / float(max(1, local_budget) * len(grounded_set)))
    else:
        alignment = 0.0

    grounded_provenance_edges = set(extract_witness_edges_in_full(Gs))
    grounded_provenance_edges |= set(delta_edges)
    grounded_provenance_edges |= set(supporting_edges)

    setattr(Gs, '_grounded_names_set', grounded_set)
    setattr(Gs, '_hit_names_set', hit_set)
    setattr(Gs, '_active_names_set', active_set)
    setattr(Gs, 'hit_constraints', sorted(hit_set))
    setattr(Gs, 'active_constraints', sorted(active_set))
    setattr(Gs, 'grounded_names', sorted(grounded_set))
    setattr(Gs, 'grounded_constraints', sorted(grounded_set))
    setattr(Gs, 'covered_constraints', sorted(grounded_set))
    setattr(Gs, '_grounding_details', grounded_details)
    setattr(Gs, 'delta_edges', sorted(delta_edges))
    setattr(Gs, '_delta_edges', sorted(delta_edges))
    setattr(Gs, 'supporting_edges', sorted(supporting_edges))
    setattr(Gs, '_supporting_edges', sorted(supporting_edges))
    setattr(Gs, 'grounded_provenance_edges', sorted(grounded_provenance_edges))
    setattr(Gs, '_grounded_provenance_edges', sorted(grounded_provenance_edges))
    setattr(Gs, '_repair_edges', sorted(delta_edges))
    setattr(Gs, '_rep_sum', float(len(delta_edges)))
    setattr(Gs, 'rep_sum', float(len(delta_edges)))
    setattr(Gs, '_alignment', float(max(0.0, alignment)))
    return grounded_set


def evaluate_grounding(
    Gs: Any,
    Sigma: Sequence,
    B: int,
    observed_graph: Any,
    find_pattern_matches_fn,
    backchase_repair_cost_fn,
) -> Set[str]:
    grounded_names: Set[str] = set()
    hit_names: Set[str] = set()
    active_names: Set[str] = set()
    grounded_details: List[Dict[str, Any]] = []
    delta_edges: Set[Tuple[int, int]] = set()
    supporting_edges: Set[Tuple[int, int]] = set()

    nodes_in_observed = getattr(Gs, '_nodes_in_observed', None)
    if nodes_in_observed is None:
        num_nodes = int(getattr(Gs, 'num_nodes', 0) or (Gs.x.size(0) if getattr(Gs, 'x', None) is not None else 0))
        nodes_in_observed = list(range(num_nodes))
    elif isinstance(nodes_in_observed, torch.Tensor):
        nodes_in_observed = nodes_in_observed.detach().cpu().tolist()
    else:
        nodes_in_observed = list(nodes_in_observed)

    nodes_in_full = getattr(Gs, '_nodes_in_full', None)
    if nodes_in_full is None:
        nodes_in_full = list(nodes_in_observed)
    elif isinstance(nodes_in_full, torch.Tensor):
        nodes_in_full = nodes_in_full.detach().cpu().tolist()
    else:
        nodes_in_full = list(nodes_in_full)

    observed_nodes_in_full = getattr(observed_graph, '_nodes_in_full', None)
    if observed_nodes_in_full is None:
        observed_nodes_in_full = list(range(int(getattr(observed_graph, 'num_nodes', 0) or 0)))
    elif isinstance(observed_nodes_in_full, torch.Tensor):
        observed_nodes_in_full = observed_nodes_in_full.detach().cpu().tolist()
    else:
        observed_nodes_in_full = list(observed_nodes_in_full)

    witness_nodes = set(int(n) for n in nodes_in_observed)
    witness_edges = extract_witness_edges_in_observed(Gs)
    graph_nx = _workload_graph_nx(Gs)

    def map_observed(view_node_id: int) -> int:
        return int(nodes_in_observed[int(view_node_id)])

    def observed_edge_to_full(edge: Tuple[int, int]) -> Tuple[int, int]:
        u_obs, v_obs = int(edge[0]), int(edge[1])
        if u_obs < len(observed_nodes_in_full) and v_obs < len(observed_nodes_in_full):
            return canonical_edge(int(observed_nodes_in_full[u_obs]), int(observed_nodes_in_full[v_obs]))
        return canonical_edge(u_obs, v_obs)

    for tgd in Sigma:
        try:
            name = tgd.get('name', 'unnamed') if isinstance(tgd, dict) else str(tgd)
        except Exception:
            name = str(tgd)

        try:
            semantics = interpret_constraint(tgd)
            matches = _consequent_matches(Gs, semantics["consequent"], find_pattern_matches_fn)
            matched_edge_count = len(semantics["consequent"].get("edges", [])) if matches else 0
        except Exception:
            matches = []
            matched_edge_count = 0

        if matches:
            hit_names.add(name)

        best_detail = None
        for bind_view in matches:
            antecedent_pattern = semantics["antecedent"]
            bind_for_antecedent = {var: node_id for var, node_id in bind_view.items() if var in antecedent_pattern.get("nodes", {})}
            assigned_count, _ = _max_node_assignment(antecedent_pattern, bind_for_antecedent, graph_nx)
            if int(assigned_count) != len(antecedent_pattern.get("nodes", {})):
                continue
            active_names.add(name)
            try:
                bind_observed = {var: map_observed(node_id) for var, node_id in bind_for_antecedent.items()}
            except Exception:
                continue

            try:
                detail = backchase_repair_cost_fn(
                    observed_graph,
                    antecedent_pattern,
                    bind_observed,
                    B,
                    witness_nodes=witness_nodes,
                    witness_edges=witness_edges,
                    return_details=True,
                )
            except Exception:
                continue

            if detail is None or not detail.get('within_budget', False):
                continue

            cost = int(detail.get('cost', B + 1))
            if cost > B:
                continue
            if best_detail is None or cost < int(best_detail.get('cost', B + 1)):
                best_detail = {
                    **detail,
                    'delta_edges_full': [observed_edge_to_full(edge) for edge in detail.get('delta_edges', [])],
                    'supporting_edges_full': [observed_edge_to_full(edge) for edge in detail.get('supporting_edges', [])],
                }

        if best_detail is None:
            continue

        grounded_names.add(name)
        grounded_details.append({
            'constraint': name,
            'antecedent_edges': len(semantics["antecedent"].get("edges", [])),
            'consequent_edges': len(semantics["consequent"].get("edges", [])),
            'target_edges': len(semantics["antecedent"].get("edges", [])) + len(semantics["consequent"].get("edges", [])),
            'activated_match_edges': int(matched_edge_count),
            'delta_edges': [tuple(edge) for edge in best_detail.get('delta_edges_full', [])],
            'supporting_edges': [tuple(edge) for edge in best_detail.get('supporting_edges_full', [])],
            'cost': int(best_detail.get('cost', 0)),
        })
        delta_edges |= {canonical_edge(*edge) for edge in best_detail.get('delta_edges_full', [])}
        supporting_edges |= {canonical_edge(*edge) for edge in best_detail.get('supporting_edges_full', [])}

    return attach_grounding_metadata(Gs, grounded_names, hit_names, active_names, grounded_details, delta_edges, supporting_edges, B)


def pair_quality(Gs: Any, conc_fn, alpha: float, beta: float) -> float:
    conc = float(conc_fn(Gs))
    alignment = float(getattr(Gs, '_alignment', 0.0))
    score = alpha * conc + beta * alignment
    setattr(Gs, '_pair_quality', score)
    return score


def window_objective(window_graphs: Sequence[Any], total_constraints: int, coverage_weight: float) -> float:
    covered: Set[str] = set()
    score_sum = 0.0
    for graph in window_graphs:
        score_sum += float(getattr(graph, '_pair_quality', 0.0))
        covered |= set(getattr(graph, '_grounded_names_set', set()))
    coverage_ratio = (len(covered) / float(total_constraints)) if total_constraints > 0 else 0.0
    return score_sum + coverage_weight * coverage_ratio


def window_coverage(window_graphs: Sequence[Any]) -> Set[str]:
    covered: Set[str] = set()
    for graph in window_graphs:
        covered |= set(getattr(graph, '_grounded_names_set', set()))
    return covered
