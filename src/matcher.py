from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import os

import networkx as nx
import torch


def _parse_edge_spec(edge: Any) -> Tuple[Any, Any, Optional[Set[Any]]]:
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


def _data_to_nx(graph_view: Any) -> nx.Graph:
    """
    Convert a PyG graph/subgraph view into an undirected NetworkX graph with:
      - node attribute `label`
      - edge attribute `rel_type` when available
    """
    graph_nx = nx.Graph()

    is_node_task = (
        getattr(graph_view, "task", None) == "node"
        or hasattr(graph_view, "_target_node_subgraph_id")
        or hasattr(graph_view, "target_node_subgraph_id")
    )

    if hasattr(graph_view, "y_type") and isinstance(graph_view.y_type, torch.Tensor):
        labels = graph_view.y_type.tolist()
        for idx, label in enumerate(labels):
            graph_nx.add_node(int(idx), label=int(label))
    elif hasattr(graph_view, "node_labels") and isinstance(graph_view.node_labels, torch.Tensor):
        labels = graph_view.node_labels.tolist()
        for idx, label in enumerate(labels):
            graph_nx.add_node(int(idx), label=int(label))
    elif is_node_task:
        raise ValueError("Node constraint matching requires explicit y_type or node_labels.")
    elif hasattr(graph_view, "x") and isinstance(graph_view.x, torch.Tensor):
        x = graph_view.x
        labels = x.argmax(dim=-1).tolist() if x.size(1) <= 10 else [0] * x.size(0)
        for idx, label in enumerate(labels):
            graph_nx.add_node(int(idx), label=int(label))
    else:
        raise ValueError("Graph view must contain x so node labels can be derived.")

    edge_rel = getattr(graph_view, "edge_rel_type", None)
    if hasattr(graph_view, "edge_index") and isinstance(graph_view.edge_index, torch.Tensor):
        edge_index = graph_view.edge_index
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must have shape [2, E]"
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        rel_values = None
        if isinstance(edge_rel, torch.Tensor) and edge_rel.numel() == edge_index.size(1):
            rel_values = edge_rel.detach().cpu().tolist()
        for pos, (u, v) in enumerate(zip(src, dst)):
            if u == v:
                continue
            attrs: Dict[str, Any] = {}
            if rel_values is not None:
                attrs["rel_type"] = rel_values[pos]
            graph_nx.add_edge(int(u), int(v), **attrs)
    elif hasattr(graph_view, "edges") and isinstance(graph_view.edges, list):
        for u, v in graph_view.edges:
            if u == v:
                continue
            graph_nx.add_edge(int(u), int(v))
    else:
        raise ValueError("Graph view needs edge_index or edges.")

    return graph_nx


def _build_pattern(pattern_spec: Dict[str, Any]) -> nx.Graph:
    pattern = nx.Graph()
    nodes: Dict[str, Dict[str, Any]] = pattern_spec.get("nodes", {})
    edges: Iterable[Tuple[str, str]] = pattern_spec.get("edges", [])

    for var, cond in nodes.items():
        allowed = set(cond.get("in", []))
        pattern.add_node(var, allowed=allowed)
    for edge in edges:
        u, v, rel_allowed = _parse_edge_spec(edge)
        if u == v:
            continue
        pattern.add_edge(u, v, allowed_rel=rel_allowed)
    return pattern


def _node_match(attrs_graph: Dict[str, Any], attrs_pattern: Dict[str, Any]) -> bool:
    label = attrs_graph.get("label", None)
    allowed = attrs_pattern.get("allowed", None)
    if allowed is None:
        return True
    if label is None:
        return False
    return label in allowed


def _edge_match(attrs_graph: Dict[str, Any], attrs_pattern: Dict[str, Any]) -> bool:
    allowed_rel = attrs_pattern.get("allowed_rel", None)
    if allowed_rel is None:
        return True
    rel_type = attrs_graph.get("rel_type", None)
    return rel_type in allowed_rel


def iter_pattern_matches(
    graph_view: Any,
    pattern_spec: Dict[str, Any],
    max_results: Optional[int] = None,
) -> Iterable[Dict[str, int]]:
    """
    Find all VF2 subgraph matches of a pattern specification in the given graph view.
    Each result is a dict mapping pattern variables to node ids in the graph view.
    """
    graph_nx = _data_to_nx(graph_view)
    pattern = _build_pattern(pattern_spec)

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph_nx,
        pattern,
        node_match=lambda ng, np: _node_match(ng, np),
        edge_match=lambda eg, ep: _edge_match(eg, ep),
    )

    emitted = 0
    for mapping in matcher.subgraph_isomorphisms_iter():
        inverse: Dict[str, int] = {}
        for node_graph, node_pattern in mapping.items():
            inverse[node_pattern] = int(node_graph)
        distinct = pattern_spec.get("distinct", [])
        if distinct:
            bound_nodes = [inverse[var] for var in distinct if var in inverse]
            if len(bound_nodes) != len(set(bound_nodes)):
                continue
        yield inverse
        emitted += 1
        if max_results is not None and emitted >= int(max_results):
            break


def find_pattern_matches(
    graph_view: Any,
    pattern_spec: Dict[str, Any],
    max_results: Optional[int] = None,
) -> List[Dict[str, int]]:
    results: List[Dict[str, int]] = list(iter_pattern_matches(graph_view, pattern_spec, max_results=max_results))
    return results


def match_consequent_instances(graph_view: Any, constraint: Dict[str, Any]) -> List[Dict[str, int]]:
    consequent = constraint.get("consequent", {})
    if not consequent.get("edges", []):
        return []
    return find_pattern_matches(graph_view, consequent)


def _edge_exists(graph_nx: nx.Graph, u: int, v: int, allowed_rel: Optional[Set[Any]] = None) -> bool:
    if not (graph_nx.has_edge(u, v) or graph_nx.has_edge(v, u)):
        return False
    if allowed_rel is None:
        return True
    attrs = graph_nx.get_edge_data(u, v, default=None)
    if attrs is None:
        attrs = graph_nx.get_edge_data(v, u, default=None)
    rel_type = None if attrs is None else attrs.get("rel_type", None)
    return rel_type in allowed_rel


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _nodes_of_label(graph_nx: nx.Graph, allowed: Set[int]) -> List[int]:
    out: List[int] = []
    for node_id, attrs in graph_nx.nodes(data=True):
        if attrs.get("label", None) in allowed:
            out.append(int(node_id))
    return out


def backchase_repair_cost(
    graph_view: Any,
    antecedent_pattern: Dict[str, Any],
    binding: Dict[str, int],
    B: int,
    witness_nodes: Optional[Set[int]] = None,
    witness_edges: Optional[Set[Tuple[int, int]]] = None,
    return_details: bool = False,
) -> Tuple[bool, int, List[Tuple[int, int]]]:
    """
    Given a consequent match binding, estimate the minimum missing-edge cost needed
    to satisfy P (antecedent) inside the current witness node set.

    Semantics:
      - supporting edges exist in the observed graph but not in G_s and cost 0
      - grounded edges do not exist in the observed graph and count toward ΔE
      - no new nodes may be introduced; all bindings must stay inside witness_nodes
    """
    graph_nx = _data_to_nx(graph_view)
    antecedent_nodes: Dict[str, Dict[str, Any]] = antecedent_pattern.get("nodes", {})
    antecedent_edges = [_parse_edge_spec(edge) for edge in list(antecedent_pattern.get("edges", []))]
    distinct_vars = antecedent_pattern.get("distinct", [])
    witness_edges = {_canonical_edge(u, v) for (u, v) in (witness_edges or set())}
    debug_witness = os.environ.get("DEBUG_MATCHER", "") == "1"

    def _distinct_blocked(var_name: str, env: Dict[str, int]) -> Set[int]:
        if not distinct_vars or var_name not in distinct_vars:
            return set()
        return {env[dv] for dv in distinct_vars if dv in env and dv != var_name}

    allowed_map: Dict[str, Set[int]] = {
        var: set(cond.get("in", [])) for var, cond in antecedent_nodes.items()
    }

    def _ordered_candidates(
        var_name: str,
        neighbor_of: Optional[int],
        env: Dict[str, int],
        allowed_rel: Optional[Set[Any]] = None,
    ) -> List[int]:
        allowed = allowed_map.get(var_name, set())
        blocked = _distinct_blocked(var_name, env)
        all_nodes = [n for n in _nodes_of_label(graph_nx, allowed) if n not in blocked]
        if witness_nodes is not None:
            ordered = [n for n in all_nodes if n in witness_nodes]
        else:
            ordered = all_nodes
        if neighbor_of is not None:
            ordered = sorted(
                ordered,
                key=lambda n: (0 if _edge_exists(graph_nx, neighbor_of, n, allowed_rel) else 1, n),
            )
        return ordered

    best: Optional[Dict[str, Any]] = None

    def _update_best(
        env: Dict[str, int],
        delta_edges: List[Tuple[int, int]],
        supporting_edges: Set[Tuple[int, int]],
    ) -> None:
        nonlocal best
        delta_cost = len({_canonical_edge(u, v) for (u, v) in delta_edges})
        candidate = {
            "env": dict(env),
            "delta_edges": [_canonical_edge(u, v) for (u, v) in delta_edges],
            "supporting_edges": sorted(supporting_edges),
            "cost": delta_cost,
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    def _edge_contribution(
        u_id: int,
        v_id: int,
        allowed_rel: Optional[Set[Any]],
    ) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]]]:
        edge = _canonical_edge(u_id, v_id)
        if _edge_exists(graph_nx, u_id, v_id, allowed_rel):
            if edge in witness_edges:
                return [], set()
            return [], {edge}
        return [edge], set()

    def _search(
        edge_pos: int,
        env: Dict[str, int],
        delta_edges: List[Tuple[int, int]],
        supporting_edges: Set[Tuple[int, int]],
    ) -> None:
        if len({_canonical_edge(u, v) for (u, v) in delta_edges}) > B:
            return
        if edge_pos >= len(antecedent_edges):
            _update_best(env, delta_edges, supporting_edges)
            return

        u_var, v_var, allowed_rel = antecedent_edges[edge_pos]
        u_bound = u_var in env
        v_bound = v_var in env

        if debug_witness:
            print(
                f"[MATCHER DEBUG] Processing antecedent edge ({u_var}, {v_var}): "
                f"u_bound={u_bound}, v_bound={v_bound}, env={env}"
            )

        if u_bound and v_bound:
            extra_delta, extra_support = _edge_contribution(env[u_var], env[v_var], allowed_rel)
            _search(edge_pos + 1, env, delta_edges + extra_delta, supporting_edges | extra_support)
            return

        if u_bound and not v_bound:
            u_id = env[u_var]
            for cand in _ordered_candidates(v_var, neighbor_of=u_id, env=env, allowed_rel=allowed_rel):
                new_env = dict(env)
                new_env[v_var] = cand
                extra_delta, extra_support = _edge_contribution(u_id, cand, allowed_rel)
                _search(edge_pos + 1, new_env, delta_edges + extra_delta, supporting_edges | extra_support)
            return

        if not u_bound and v_bound:
            v_id = env[v_var]
            for cand in _ordered_candidates(u_var, neighbor_of=v_id, env=env, allowed_rel=allowed_rel):
                new_env = dict(env)
                new_env[u_var] = cand
                extra_delta, extra_support = _edge_contribution(cand, v_id, allowed_rel)
                _search(edge_pos + 1, new_env, delta_edges + extra_delta, supporting_edges | extra_support)
            return

        u_candidates = _ordered_candidates(u_var, neighbor_of=None, env=env, allowed_rel=allowed_rel)
        v_candidates = _ordered_candidates(v_var, neighbor_of=None, env=env, allowed_rel=allowed_rel)
        seen_pairs: Set[Tuple[int, int]] = set()

        for a, b, attrs in graph_nx.edges(data=True):
            a = int(a)
            b = int(b)
            if a == b:
                continue
            if witness_nodes is not None and (a not in witness_nodes or b not in witness_nodes):
                continue
            if allowed_rel is not None and attrs.get("rel_type", None) not in allowed_rel:
                continue
            if a in _distinct_blocked(u_var, env) or b in _distinct_blocked(v_var, env):
                continue
            if graph_nx.nodes[a].get("label", None) in allowed_map.get(u_var, set()) and graph_nx.nodes[b].get("label", None) in allowed_map.get(v_var, set()):
                seen_pairs.add((a, b))
            if graph_nx.nodes[a].get("label", None) in allowed_map.get(v_var, set()) and graph_nx.nodes[b].get("label", None) in allowed_map.get(u_var, set()):
                seen_pairs.add((b, a))

        for a, b in list(seen_pairs):
            if a == b and u_var in distinct_vars and v_var in distinct_vars:
                continue
            new_env = dict(env)
            new_env[u_var] = a
            new_env[v_var] = b
            extra_delta, extra_support = _edge_contribution(a, b, allowed_rel)
            _search(edge_pos + 1, new_env, delta_edges + extra_delta, supporting_edges | extra_support)

        max_pair_nodes = 32
        for a in u_candidates[:max_pair_nodes]:
            for b in v_candidates[:max_pair_nodes]:
                if a == b and (u_var in distinct_vars or v_var in distinct_vars):
                    continue
                if (a, b) in seen_pairs:
                    continue
                new_env = dict(env)
                new_env[u_var] = a
                new_env[v_var] = b
                extra_delta, extra_support = _edge_contribution(a, b, allowed_rel)
                _search(edge_pos + 1, new_env, delta_edges + extra_delta, supporting_edges | extra_support)

    _search(0, dict(binding), [], set())

    if best is None:
        result = {
            "within_budget": False,
            "cost": B + 1,
            "repairs": [],
            "delta_edges": [],
            "supporting_edges": [],
            "antecedent_binding": dict(binding),
        }
    else:
        result = {
            "within_budget": best["cost"] <= B,
            "cost": int(best["cost"]),
            "repairs": list(best["delta_edges"]),
            "delta_edges": list(best["delta_edges"]),
            "supporting_edges": list(best["supporting_edges"]),
            "antecedent_binding": best["env"],
        }

    if return_details:
        return result
    return (result["within_budget"], result["cost"], result["repairs"])


class MatchResult:
    def __init__(self, grounded: bool, rep_cost: int, repairs: List[Tuple[int, int]]):
        self.grounded = grounded
        self.rep_cost = rep_cost
        self.repairs = repairs
