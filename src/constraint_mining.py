from __future__ import annotations

import json
import os
import random
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

try:
    from constraints import get_constraints, validate_tgd
except ImportError:
    from .constraints import get_constraints, validate_tgd


TGD = Dict[str, Any]
ROOT_DIR = Path(__file__).resolve().parents[1]


def _parse_edge_spec(edge: Any) -> Tuple[Any, Any, Any]:
    if not isinstance(edge, (tuple, list)):
        raise ValueError(f"Unsupported edge spec: {edge}")
    if len(edge) == 2:
        return edge[0], edge[1], None
    if len(edge) == 3:
        return edge[0], edge[1], edge[2]
    raise ValueError(f"Unsupported edge spec length: {edge}")


def _is_graph_dataset_resource(dataset_resource: Any) -> bool:
    return isinstance(dataset_resource, dict) and 'train_loader' in dataset_resource and 'dataset' in dataset_resource


def _is_node_dataset_resource(dataset_resource: Any) -> bool:
    return (isinstance(dataset_resource, dict) and 'data' in dataset_resource) or isinstance(dataset_resource, Data)


def _clone_constraints(constraints: Sequence[TGD]) -> List[TGD]:
    return json.loads(json.dumps(list(constraints)))


def _builtin_constraint_pool_path(dataset_key: str, mode: str) -> Optional[Path]:
    mapping = {
        ("dblp", "balanced"): ROOT_DIR / "configs" / "constraint_pools" / "dblp_balanced_p7_t0p4.yaml",
        ("dblp", "coverage_only"): ROOT_DIR / "configs" / "constraint_pools" / "dblp_coverage_only.yaml",
    }
    return mapping.get((str(dataset_key).lower(), str(mode).lower()))


def _load_constraint_pool_names(pool_path: Path) -> List[str]:
    suffix = pool_path.suffix.lower()
    if suffix == ".csv":
        with open(pool_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            names: List[str] = []
            for row in reader:
                selected = str(row.get("selected", "true")).strip().lower()
                if selected in {"false", "0", "no"}:
                    continue
                name = str(row.get("constraint_name", "")).strip()
                if name:
                    names.append(name)
            return names

    if suffix == ".json":
        with open(pool_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"YAML support is required to read constraint pool file: {pool_path}") from exc
        with open(pool_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

    if isinstance(payload, dict):
        names = payload.get("constraint_names", [])
    elif isinstance(payload, list):
        names = payload
    else:
        names = []
    return [str(name).strip() for name in names if str(name).strip()]


def _apply_constraint_pool_mode(
    constraints: Sequence[TGD],
    config: Dict[str, Any],
    dataset_key: str,
) -> List[TGD]:
    mode = str(config.get("constraint_pool_mode", "original")).strip().lower()
    if mode in {"", "original", "default"}:
        config["_resolved_constraint_pool_mode"] = "original"
        return list(constraints)

    pool_file_value = config.get("constraint_pool_file")
    pool_path = Path(str(pool_file_value)).expanduser() if pool_file_value else _builtin_constraint_pool_path(dataset_key, mode)
    if pool_path is None:
        raise ValueError(
            f"Unsupported constraint_pool_mode='{mode}' for dataset '{dataset_key}'. "
            "Use original or provide constraint_pool_file."
        )
    if not pool_path.is_absolute():
        pool_path = (ROOT_DIR / pool_path).resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"Constraint pool file not found: {pool_path}")

    names = _load_constraint_pool_names(pool_path)
    by_name = {str(tgd.get("name", "")): tgd for tgd in constraints}
    selected: List[TGD] = []
    missing: List[str] = []
    for name in names:
        tgd = by_name.get(name)
        if tgd is None:
            missing.append(name)
            continue
        selected.append(tgd)

    if missing:
        raise KeyError(
            f"Constraint pool file '{pool_path}' referenced missing rules: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )

    config["_resolved_constraint_pool_mode"] = mode
    config["_resolved_constraint_pool_file"] = str(pool_path)
    return selected


def _cache_dir(config: Dict[str, Any]) -> str:
    path = config.get("constraint_cache_dir")
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    path = os.path.join("artifacts", "constraints")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(config: Dict[str, Any], dataset_key: str) -> str:
    source = str(config.get("constraint_source", "static")).lower()
    type_source = str(config.get("constraint_type_source", "auto")).lower()
    rule_mode = str(config.get("constraint_rule_mode", "legacy")).lower()
    l_hops = int(config.get("constraint_mining_hops", config.get("L", 2)))
    support = int(config.get("constraint_min_support", 3))
    confidence = float(config.get("constraint_min_confidence", 0.6))
    max_patterns = int(config.get("constraint_max_patterns", 8))
    seed = int(config.get("random_seed", 0))
    templates = config.get("constraint_templates", ["triangle", "extension"])
    if isinstance(templates, (list, tuple, set)):
        template_token = "-".join(sorted(str(t).lower() for t in templates))
    else:
        template_token = str(templates).lower()
    target_match_flag = 1 if bool(config.get("constraint_filter_target_matchability", False)) else 0
    filename = (
        f"{dataset_key.lower()}_{source}_{type_source}"
        f"_{rule_mode}_L{l_hops}_sup{support}_conf{confidence:.2f}_max{max_patterns}"
        f"_tmpl_{template_token}_tm_{target_match_flag}_seed{seed}.json"
    )
    return os.path.join(_cache_dir(config), filename)


def _type_cache_dir(config: Dict[str, Any]) -> str:
    path = os.path.join(_cache_dir(config), "types")
    os.makedirs(path, exist_ok=True)
    return path


def _degree_bucket_labels(data: Data, num_buckets: int) -> torch.Tensor:
    row = data.edge_index[0].detach().cpu()
    degrees = torch.bincount(row, minlength=data.num_nodes).to(torch.float)
    if int(degrees.numel()) == 0:
        return torch.zeros(data.num_nodes, dtype=torch.long)
    quantiles = torch.linspace(0.0, 1.0, steps=max(2, num_buckets + 1))
    boundaries = torch.quantile(degrees, quantiles).tolist()
    labels = torch.zeros(data.num_nodes, dtype=torch.long)
    for idx, value in enumerate(degrees.tolist()):
        bucket = 0
        while bucket + 1 < len(boundaries) - 1 and value > boundaries[bucket + 1]:
            bucket += 1
        labels[idx] = int(bucket)
    return labels


def _pseudo_type_cache_path(config: Dict[str, Any], data_name: str, num_types: int, seed: int, suffix: str) -> str:
    safe_suffix = suffix.replace("/", "_").replace(" ", "_").lower()
    return os.path.join(_type_cache_dir(config), f"{data_name}_{safe_suffix}_k{num_types}_seed{seed}.pt")


def _cluster_pseudo_type_labels(config: Dict[str, Any], data: Data, num_types: int, seed: int, cache_suffix: str) -> torch.Tensor:
    data_name = str(config.get("data_name", "graph")).lower()
    cache_path = _pseudo_type_cache_path(config, data_name, num_types, seed, cache_suffix)
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    fallback = _degree_bucket_labels(data, num_types)
    x = getattr(data, "x", None)
    train_mask = getattr(data, "train_mask", None)
    if x is None or not isinstance(x, torch.Tensor) or x.dim() != 2:
        torch.save(fallback, cache_path)
        return fallback

    if train_mask is None or int(train_mask.sum()) == 0:
        train_x = x.detach().cpu().numpy()
    else:
        train_x = x[train_mask].detach().cpu().numpy()
    full_x = x.detach().cpu().numpy()

    if train_x.shape[0] < max(2, num_types):
        torch.save(fallback, cache_path)
        return fallback

    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_types, random_state=seed, n_init=10)
        kmeans.fit(train_x)
        labels = torch.tensor(kmeans.predict(full_x), dtype=torch.long)
        torch.save(labels, cache_path)
        return labels
    except Exception:
        torch.save(fallback, cache_path)
        return fallback


def _pseudo_type_labels(config: Dict[str, Any], data: Data) -> torch.Tensor:
    data_name = str(config.get("data_name", "graph")).lower()
    seed = int(config.get("random_seed", 0))
    num_types = int(config.get("n_node_types", 8))
    type_source = str(config.get("constraint_type_source", "pseudo_type")).lower()

    if type_source in {"degree_bucket", "degree_buckets"}:
        cache_path = _pseudo_type_cache_path(config, data_name, num_types, seed, "degree_bucket")
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        labels = _degree_bucket_labels(data, num_types)
        torch.save(labels, cache_path)
        return labels

    if type_source in {"pseudo_type_degree_combo", "pseudo_type_combo", "feature_degree_combo"}:
        combo_cache = _pseudo_type_cache_path(config, data_name, num_types, seed, "pseudo_type_degree_combo")
        if os.path.exists(combo_cache):
            return torch.load(combo_cache)

        cluster_count = max(2, int(config.get("pseudo_cluster_types", num_types)))
        degree_buckets = max(2, int(config.get("pseudo_degree_buckets", min(4, num_types))))
        cluster_labels = _cluster_pseudo_type_labels(config, data, cluster_count, seed, "pseudo_cluster_base")
        degree_labels = _degree_bucket_labels(data, degree_buckets)

        train_mask = getattr(data, "train_mask", None)
        train_indices = (
            torch.where(train_mask)[0].detach().cpu().tolist()
            if isinstance(train_mask, torch.Tensor) and int(train_mask.sum()) > 0
            else list(range(int(data.num_nodes)))
        )

        combo_counts: Counter[Tuple[int, int]] = Counter(
            (int(cluster_labels[idx]), int(degree_labels[idx])) for idx in train_indices
        )
        top_combos = [combo for combo, _ in combo_counts.most_common(max(1, num_types))]
        combo_to_id = {combo: idx for idx, combo in enumerate(top_combos)}

        labels = torch.zeros(int(data.num_nodes), dtype=torch.long)
        for idx in range(int(data.num_nodes)):
            combo = (int(cluster_labels[idx]), int(degree_labels[idx]))
            if combo in combo_to_id:
                labels[idx] = int(combo_to_id[combo])
            else:
                labels[idx] = int(int(cluster_labels[idx]) % max(1, num_types))

        torch.save(labels, combo_cache)
        return labels

    return _cluster_pseudo_type_labels(config, data, num_types, seed, "pseudo_types")


def _dblp_native_bucket_types(config: Dict[str, Any], data: Data) -> torch.Tensor:
    native = getattr(data, "native_node_type", None)
    if not isinstance(native, torch.Tensor) or native.numel() != data.num_nodes:
        native = getattr(data, "y_type", None)
    if not isinstance(native, torch.Tensor) or native.numel() != data.num_nodes:
        raise ValueError("DBLP typed constraint mining requires native_node_type or y_type on the homogeneous view.")

    edge_index = data.edge_index.detach().cpu()
    degrees = torch.bincount(edge_index[0], minlength=data.num_nodes).to(torch.float)
    native = native.detach().cpu().to(torch.long)
    labels = torch.zeros(int(data.num_nodes), dtype=torch.long)

    author_buckets = max(4, int(config.get("dblp_author_degree_buckets", 4)))
    paper_buckets = max(6, int(config.get("dblp_paper_degree_buckets", 6)))
    term_buckets = max(10, int(config.get("dblp_term_frequency_buckets", 10)))
    conference_mode = str(config.get("dblp_conference_type_mode", "identity")).lower()

    next_id = 0
    type_groups: Dict[str, List[int]] = {}

    def _assign_degree_buckets(mask: torch.Tensor, buckets: int, group_name: str) -> None:
        nonlocal next_id
        idx = torch.where(mask)[0]
        vals = degrees[idx]
        if idx.numel() == 0:
            type_groups[group_name] = []
            return
        if idx.numel() <= buckets:
            buckets = int(idx.numel())
        boundaries = torch.quantile(vals, torch.linspace(0.0, 1.0, steps=buckets + 1)).tolist()
        type_groups[group_name] = list(range(next_id, next_id + buckets))
        for node_id, value in zip(idx.tolist(), vals.tolist()):
            bucket = 0
            while bucket + 1 < len(boundaries) - 1 and value > boundaries[bucket + 1]:
                bucket += 1
            labels[int(node_id)] = int(next_id + bucket)
        next_id += buckets

    _assign_degree_buckets(native == 0, author_buckets, "author")
    _assign_degree_buckets(native == 1, paper_buckets, "paper")
    _assign_degree_buckets(native == 2, term_buckets, "term")

    conf_idx = torch.where(native == 3)[0]
    if conf_idx.numel() == 0:
        type_groups["conference"] = []
    elif conference_mode == "identity":
        type_groups["conference"] = list(range(next_id, next_id + int(conf_idx.numel())))
        for offset, node_id in enumerate(conf_idx.tolist()):
            labels[int(node_id)] = int(next_id + offset)
        next_id += int(conf_idx.numel())
    else:
        conf_buckets = max(4, int(config.get("dblp_conference_buckets", 6)))
        _assign_degree_buckets(native == 3, conf_buckets, "conference")

    data.constraint_type_groups = {group: list(ids) for group, ids in type_groups.items()}
    data.constraint_type_native = native.clone()
    return labels


def _node_model_outputs(model: torch.nn.Module, data: Data, device: torch.device) -> Optional[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        try:
            out = model(data.x.to(device), data.edge_index.to(device))
            return out.detach().cpu()
        except Exception:
            return None


def prepare_constraint_types(
    config: Dict[str, Any],
    dataset_resource: Any,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Attach reproducible pseudo node types for constraint matching/mining."""
    if not _is_node_dataset_resource(dataset_resource):
        return

    data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource

    type_source = str(config.get("constraint_type_source", "pseudo_type")).lower()

    if type_source in {"dblp_native_bucket", "native_bucket"} and str(config.get("data_name", "")).upper() == "DBLP":
        data.y_type = _dblp_native_bucket_types(config, data)
        return

    if hasattr(data, "y_type") and data.y_type is not None:
        return

    if type_source in {
        "pseudo_type",
        "pseudo",
        "feature_cluster",
        "feature_space_clustering",
        "pseudo_type_degree_combo",
        "pseudo_type_combo",
        "feature_degree_combo",
        "degree_bucket",
        "degree_buckets",
    }:
        data.y_type = _pseudo_type_labels(config, data)
        return

    if type_source in {"predicted", "predicted_label", "model_prediction"} and model is not None:
        dev = device or next(model.parameters()).device
        out = _node_model_outputs(model, data, dev)
        if out is not None:
            if out.dim() == 2:
                data.y_type = out.argmax(dim=-1).to(torch.long)
            elif out.dim() == 1:
                data.y_type = (out > 0).to(torch.long)
            if hasattr(data, "y_type"):
                return

    if type_source in {"ground_truth", "label", "labels"}:
        if hasattr(data, "y") and isinstance(data.y, torch.Tensor) and data.y.dim() == 1 and data.y.numel() == data.num_nodes:
            data.y_type = data.y.detach().cpu().to(torch.long)
            return

    if type_source in {"feature_argmax", "auto", "predicted", "predicted_label", "model_prediction"}:
        if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
            if data.x.dim() == 2 and data.x.size(1) > 0:
                if data.x.size(1) <= 64:
                    data.y_type = data.x.argmax(dim=-1).detach().cpu().to(torch.long)
                else:
                    data.y_type = torch.zeros(data.num_nodes, dtype=torch.long)
                return

    if hasattr(data, "y") and isinstance(data.y, torch.Tensor) and data.y.dim() == 1 and data.y.numel() == data.num_nodes:
        data.y_type = data.y.detach().cpu().to(torch.long)


def _node_type_vector(graph: Data) -> List[int]:
    if hasattr(graph, "y_type") and isinstance(graph.y_type, torch.Tensor) and graph.y_type.numel() == graph.num_nodes:
        return [int(v) for v in graph.y_type.detach().cpu().tolist()]
    if hasattr(graph, "node_labels") and isinstance(graph.node_labels, torch.Tensor) and graph.node_labels.numel() == graph.num_nodes:
        return [int(v) for v in graph.node_labels.detach().cpu().tolist()]
    if hasattr(graph, "y") and isinstance(graph.y, torch.Tensor) and graph.y.dim() == 1 and graph.y.numel() == graph.num_nodes:
        return [int(v) for v in graph.y.detach().cpu().tolist()]
    if hasattr(graph, "x") and isinstance(graph.x, torch.Tensor) and graph.x.dim() == 2 and graph.x.size(1) > 0:
        if graph.x.size(1) <= 64:
            return [int(v) for v in graph.x.argmax(dim=-1).detach().cpu().tolist()]
    return [0 for _ in range(int(graph.num_nodes))]


def _native_type_vector(graph: Data) -> List[int]:
    if hasattr(graph, "constraint_type_native") and isinstance(graph.constraint_type_native, torch.Tensor):
        return [int(v) for v in graph.constraint_type_native.detach().cpu().tolist()]
    if hasattr(graph, "native_node_type") and isinstance(graph.native_node_type, torch.Tensor):
        return [int(v) for v in graph.native_node_type.detach().cpu().tolist()]
    return _node_type_vector(graph)


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _adjacency(graph: Data) -> Tuple[List[int], Dict[int, List[int]], List[Tuple[int, int]]]:
    labels = _node_type_vector(graph)
    neighbors: Dict[int, set] = {idx: set() for idx in range(int(graph.num_nodes))}
    edge_set = set()
    edge_index = graph.edge_index.detach().cpu()
    for col in range(edge_index.size(1)):
        u = int(edge_index[0, col])
        v = int(edge_index[1, col])
        if u == v:
            continue
        edge = _canonical_edge(u, v)
        edge_set.add(edge)
        neighbors[u].add(v)
        neighbors[v].add(u)
    return labels, {k: sorted(v) for k, v in neighbors.items()}, sorted(edge_set)


def _rel_adjacency(graph: Data) -> Dict[int, List[Tuple[int, int]]]:
    neighbors: Dict[int, List[Tuple[int, int]]] = {idx: [] for idx in range(int(graph.num_nodes))}
    rel_type = getattr(graph, "edge_rel_type", None)
    if not isinstance(rel_type, torch.Tensor) or rel_type.numel() != graph.edge_index.size(1):
        raise ValueError("Relation-aware mining requires edge_rel_type on the graph view.")
    rel_vals = rel_type.detach().cpu().tolist()
    edge_index = graph.edge_index.detach().cpu()
    seen: Set[Tuple[int, int, int]] = set()
    for col in range(edge_index.size(1)):
        u = int(edge_index[0, col])
        v = int(edge_index[1, col])
        r = int(rel_vals[col])
        if u == v:
            continue
        key = (min(u, v), max(u, v), r)
        if key in seen:
            continue
        seen.add(key)
        neighbors[u].append((v, r))
        neighbors[v].append((u, r))
    return neighbors


def _iter_training_samples(dataset_resource: Any, config: Dict[str, Any]) -> Iterable[Data]:
    rng = random.Random(int(config.get("random_seed", 0)))
    max_samples = int(config.get("constraint_mining_max_samples", 64))
    num_hops = int(config.get("constraint_mining_hops", config.get("L", 2)))

    if _is_graph_dataset_resource(dataset_resource):
        train_subset = dataset_resource["train_loader"].dataset
        dataset = dataset_resource["dataset"]
        if hasattr(train_subset, "indices"):
            indices = list(train_subset.indices)
        else:
            indices = list(range(len(train_subset)))
        if len(indices) > max_samples:
            indices = sorted(rng.sample(indices, max_samples))
        for idx in indices:
            graph = dataset[int(idx)].clone()
            graph.num_nodes = int(graph.num_nodes if graph.num_nodes is not None else graph.x.size(0))
            yield graph
        return

    data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
    train_nodes = torch.where(data.train_mask)[0].detach().cpu().tolist()
    if len(train_nodes) > max_samples:
        train_nodes = sorted(rng.sample(train_nodes, max_samples))
    for node_id in train_nodes:
        subset, edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=int(node_id),
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        subgraph = Data(
            x=data.x[subset],
            edge_index=edge_index,
            num_nodes=int(subset.numel()),
        )
        if hasattr(data, "y_type") and data.y_type is not None:
            subgraph.y_type = data.y_type[subset]
        elif hasattr(data, "node_labels") and data.node_labels is not None:
            subgraph.node_labels = data.node_labels[subset]
        elif hasattr(data, "y") and data.y is not None and data.y.dim() == 1:
            subgraph.y = data.y[subset]
        if hasattr(data, "native_node_type") and data.native_node_type is not None:
            subgraph.native_node_type = data.native_node_type[subset]
        if hasattr(data, "constraint_type_native") and data.constraint_type_native is not None:
            subgraph.constraint_type_native = data.constraint_type_native[subset]
        if hasattr(data, "edge_rel_type") and data.edge_rel_type is not None:
            subgraph.edge_rel_type = data.edge_rel_type[edge_mask]
        yield subgraph


def _dblp_constraint_groups(dataset_resource: Any) -> Dict[str, List[int]]:
    data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
    groups = getattr(data, "constraint_type_groups", None)
    if isinstance(groups, dict) and groups:
        return {str(k): [int(v) for v in vals] for k, vals in groups.items()}
    native = _native_type_vector(data)
    subtype = _node_type_vector(data)
    out: Dict[str, Set[int]] = {"author": set(), "paper": set(), "term": set(), "conference": set()}
    for nat, sub in zip(native, subtype):
        if nat == 0:
            out["author"].add(int(sub))
        elif nat == 1:
            out["paper"].add(int(sub))
        elif nat == 2:
            out["term"].add(int(sub))
        elif nat == 3:
            out["conference"].add(int(sub))
    return {k: sorted(v) for k, v in out.items()}


def _pattern_nodes_for_edges(nodes_spec: Dict[str, Any], edges: Sequence[Tuple[Any, ...]]) -> Dict[str, Any]:
    used: Dict[str, Any] = {}
    for edge in edges:
        u, v, _ = _parse_edge_spec(edge)
        if str(u) in nodes_spec:
            used[str(u)] = dict(nodes_spec[str(u)])
        if str(v) in nodes_spec:
            used[str(v)] = dict(nodes_spec[str(v)])
    return used


def _dedupe_tgds(constraints: Sequence[TGD]) -> List[TGD]:
    unique: List[TGD] = []
    seen: set[str] = set()
    for tgd in constraints:
        signature = json.dumps(
            {
                "consequent": tgd.get("consequent", {}),
                "antecedent": tgd.get("antecedent", {}),
            },
            sort_keys=True,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(tgd)
    return unique


def _iter_target_samples(dataset_resource: Any, config: Dict[str, Any]) -> Iterable[Data]:
    rng = random.Random(int(config.get("random_seed", 0)))
    max_samples = int(config.get("constraint_target_probe_samples", 64))
    num_hops = int(config.get("constraint_mining_hops", config.get("L", 2)))

    if _is_graph_dataset_resource(dataset_resource):
        test_subset = dataset_resource["test_loader"].dataset
        dataset = dataset_resource["dataset"]
        if hasattr(test_subset, "indices"):
            indices = list(test_subset.indices)
        else:
            indices = list(range(len(test_subset)))
        if len(indices) > max_samples:
            indices = sorted(rng.sample(indices, max_samples))
        for idx in indices:
            graph = dataset[int(idx)].clone()
            graph.num_nodes = int(graph.num_nodes if graph.num_nodes is not None else graph.x.size(0))
            yield graph
        return

    data = dataset_resource["data"] if isinstance(dataset_resource, dict) else dataset_resource
    if isinstance(dataset_resource, dict) and dataset_resource.get("target_nodes"):
        test_nodes = [int(v) for v in dataset_resource["target_nodes"]]
    else:
        test_nodes = torch.where(data.test_mask)[0].detach().cpu().tolist()
    if len(test_nodes) > max_samples:
        test_nodes = sorted(rng.sample(test_nodes, max_samples))
    for node_id in test_nodes:
        subset, edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=int(node_id),
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        subgraph = Data(
            x=data.x[subset],
            edge_index=edge_index,
            num_nodes=int(subset.numel()),
        )
        if hasattr(data, "y_type") and data.y_type is not None:
            subgraph.y_type = data.y_type[subset]
        elif hasattr(data, "node_labels") and data.node_labels is not None:
            subgraph.node_labels = data.node_labels[subset]
        elif hasattr(data, "y") and data.y is not None and data.y.dim() == 1:
            subgraph.y = data.y[subset]
        if hasattr(data, "native_node_type") and data.native_node_type is not None:
            subgraph.native_node_type = data.native_node_type[subset]
        if hasattr(data, "constraint_type_native") and data.constraint_type_native is not None:
            subgraph.constraint_type_native = data.constraint_type_native[subset]
        if hasattr(data, "edge_rel_type") and data.edge_rel_type is not None:
            subgraph.edge_rel_type = data.edge_rel_type[edge_mask]
        yield subgraph


def _emit_dblp_backchase_variants(
    prefix: str,
    nodes_spec: Dict[str, Any],
    motif_edges: Sequence[Tuple[Any, ...]],
    support: int,
    template: str,
) -> List[TGD]:
    variants: List[TGD] = []
    for idx, consequent_edge in enumerate(motif_edges):
        antecedent_edges = [edge for j, edge in enumerate(motif_edges) if j != idx]
        if not antecedent_edges:
            continue
        consequent_nodes = _pattern_nodes_for_edges(nodes_spec, [consequent_edge])
        antecedent_nodes = _pattern_nodes_for_edges(nodes_spec, antecedent_edges)
        u, v, rel = _parse_edge_spec(consequent_edge)
        rel_label = "none" if rel is None else str(sorted(rel)[0] if isinstance(rel, set) and len(rel) == 1 else rel)
        tgd = {
            "name": f"{prefix}_c{idx}_{str(u).lower()}_{str(v).lower()}_{rel_label}",
            "consequent": {
                "nodes": consequent_nodes,
                "edges": [consequent_edge],
                "distinct": sorted(consequent_nodes.keys()),
            },
            "antecedent": {
                "nodes": antecedent_nodes,
                "edges": list(antecedent_edges),
                "distinct": sorted(antecedent_nodes.keys()),
            },
            "mining": {
                "template": template,
                "support": int(support),
                "confidence": 1.0,
                "constraint_rule_mode": "standard_backchase",
                "consequent_edge": consequent_edge,
                "antecedent_edge_count": int(len(antecedent_edges)),
            },
        }
        validate_tgd(tgd)
        variants.append(tgd)
    return variants


def _filter_dblp_constraints_by_consequent_matchability(
    constraints: Sequence[TGD],
    dataset_resource: Any,
    config: Dict[str, Any],
) -> List[TGD]:
    if not constraints:
        return []
    try:
        from matcher import find_pattern_matches
    except ImportError:
        from .matcher import find_pattern_matches

    workload_samples = list(_iter_target_samples(dataset_resource, config))
    min_hits = int(config.get("constraint_target_min_hit", 1))
    audited: List[TGD] = []
    for tgd in constraints:
        workload_hit_count = 0
        consequent_match_count = 0
        for sample in workload_samples:
            try:
                matches = find_pattern_matches(sample, tgd.get("consequent", {}))
            except Exception:
                matches = []
            if matches:
                workload_hit_count += 1
                consequent_match_count += int(len(matches))
        enriched = json.loads(json.dumps(tgd))
        mining = dict(enriched.get("mining", {}))
        mining["target_workload_hit_count"] = int(workload_hit_count)
        mining["target_consequent_match_count"] = int(consequent_match_count)
        mining["target_probe_sample_count"] = int(len(workload_samples))
        enriched["mining"] = mining
        audited.append(enriched)

    audited.sort(
        key=lambda tgd: (
            -int(tgd.get("mining", {}).get("target_workload_hit_count", 0)),
            -int(tgd.get("mining", {}).get("support", 0)),
            str(tgd.get("mining", {}).get("template", "")),
            str(tgd.get("name", "")),
        )
    )
    filtered = [tgd for tgd in audited if int(tgd.get("mining", {}).get("target_workload_hit_count", 0)) >= min_hits]
    if bool(config.get("constraint_filter_target_matchability", True)):
        return filtered
    return audited


def _filter_constraints_by_consequent_matchability(
    constraints: Sequence[TGD],
    dataset_resource: Any,
    config: Dict[str, Any],
) -> List[TGD]:
    if not constraints:
        return []
    try:
        from matcher import find_pattern_matches
    except ImportError:
        from .matcher import find_pattern_matches

    workload_samples = list(_iter_target_samples(dataset_resource, config))
    min_hits = int(config.get("constraint_target_min_hit", 1))
    audited: List[TGD] = []
    for tgd in constraints:
        workload_hit_count = 0
        consequent_match_count = 0
        for sample in workload_samples:
            try:
                matches = find_pattern_matches(sample, tgd.get("consequent", {}))
            except Exception:
                matches = []
            if matches:
                workload_hit_count += 1
                consequent_match_count += int(len(matches))
        enriched = json.loads(json.dumps(tgd))
        mining = dict(enriched.get("mining", {}))
        mining["target_workload_hit_count"] = int(workload_hit_count)
        mining["target_consequent_match_count"] = int(consequent_match_count)
        mining["target_probe_sample_count"] = int(len(workload_samples))
        enriched["mining"] = mining
        audited.append(enriched)

    audited.sort(
        key=lambda tgd: (
            -int(tgd.get("mining", {}).get("target_workload_hit_count", 0)),
            -int(tgd.get("mining", {}).get("support", 0)),
            str(tgd.get("mining", {}).get("template", "")),
            str(tgd.get("name", "")),
        )
    )
    filtered = [tgd for tgd in audited if int(tgd.get("mining", {}).get("target_workload_hit_count", 0)) >= min_hits]
    if bool(config.get("constraint_filter_target_matchability", False)):
        return filtered
    return audited


def _mine_dblp_completion_constraints(dataset_resource: Any, config: Dict[str, Any]) -> List[TGD]:
    samples = list(_iter_training_samples(dataset_resource, config))
    if not samples:
        return []

    apt_support: Counter[Tuple[int, int, int]] = Counter()
    apc_support: Counter[Tuple[int, int, int]] = Counter()
    aat_support: Counter[Tuple[int, int, int]] = Counter()
    aac_support: Counter[Tuple[int, int, int]] = Counter()

    for sample in samples:
        subtype = _node_type_vector(sample)
        native = _native_type_vector(sample)
        neighbors = _rel_adjacency(sample)

        apt_seen: Set[Tuple[int, int, int]] = set()
        apc_seen: Set[Tuple[int, int, int]] = set()
        aat_seen: Set[Tuple[int, int, int]] = set()
        aac_seen: Set[Tuple[int, int, int]] = set()

        for paper in range(int(sample.num_nodes)):
            if native[paper] != 1:
                continue
            author_neighbors = [nbr for nbr, rel in neighbors[paper] if rel == 0 and native[nbr] == 0]
            term_neighbors = [nbr for nbr, rel in neighbors[paper] if rel == 1 and native[nbr] == 2]
            conf_neighbors = [nbr for nbr, rel in neighbors[paper] if rel == 2 and native[nbr] == 3]
            for author in author_neighbors:
                for term in term_neighbors:
                    apt_seen.add((int(subtype[author]), int(subtype[paper]), int(subtype[term])))
                for conf in conf_neighbors:
                    apc_seen.add((int(subtype[author]), int(subtype[paper]), int(subtype[conf])))

        for term in range(int(sample.num_nodes)):
            if native[term] != 2:
                continue
            papers = [nbr for nbr, rel in neighbors[term] if rel == 1 and native[nbr] == 1]
            author_subtypes: Set[int] = set()
            for paper in papers:
                author_subtypes.update(int(subtype[nbr]) for nbr, rel in neighbors[paper] if rel == 0 and native[nbr] == 0)
            ordered_authors = sorted(author_subtypes)
            for i in range(len(ordered_authors)):
                for j in range(i + 1, len(ordered_authors)):
                    aat_seen.add((ordered_authors[i], int(subtype[term]), ordered_authors[j]))

        for conf in range(int(sample.num_nodes)):
            if native[conf] != 3:
                continue
            papers = [nbr for nbr, rel in neighbors[conf] if rel == 2 and native[nbr] == 1]
            author_subtypes: Set[int] = set()
            for paper in papers:
                author_subtypes.update(int(subtype[nbr]) for nbr, rel in neighbors[paper] if rel == 0 and native[nbr] == 0)
            ordered_authors = sorted(author_subtypes)
            for i in range(len(ordered_authors)):
                for j in range(i + 1, len(ordered_authors)):
                    aac_seen.add((ordered_authors[i], int(subtype[conf]), ordered_authors[j]))

        apt_support.update(apt_seen)
        apc_support.update(apc_seen)
        aat_support.update(aat_seen)
        aac_support.update(aac_seen)

    max_patterns = int(config.get("constraint_max_patterns", 64))
    max_per_template = int(config.get("dblp_max_patterns_per_template", max(8, max_patterns // 4)))
    min_support = int(config.get("constraint_min_support", 2))
    groups = _dblp_constraint_groups(dataset_resource)
    paper_allowed = groups.get("paper", [])

    def _top(counter: Counter[Tuple[int, int, int]]) -> List[Tuple[Tuple[int, int, int], int]]:
        ranked = [(key, int(support)) for key, support in counter.items() if int(support) >= min_support]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:max_per_template]

    constraints: List[TGD] = []

    for (a_sub, p_sub, t_sub), support in _top(apt_support):
        nodes_spec = {"A": {"in": [int(a_sub)]}, "P": {"in": [int(p_sub)]}, "T": {"in": [int(t_sub)]}}
        motif_edges = [("A", "P", 0), ("P", "T", 1)]
        constraints.extend(
            _emit_dblp_backchase_variants(
                prefix=f"mined_dblp_apt_{a_sub}_{p_sub}_{t_sub}",
                nodes_spec=nodes_spec,
                motif_edges=motif_edges,
                support=int(support),
                template="dblp_apt_backchase",
            )
        )

    for (a_sub, p_sub, c_sub), support in _top(apc_support):
        nodes_spec = {"A": {"in": [int(a_sub)]}, "P": {"in": [int(p_sub)]}, "C": {"in": [int(c_sub)]}}
        motif_edges = [("A", "P", 0), ("P", "C", 2)]
        constraints.extend(
            _emit_dblp_backchase_variants(
                prefix=f"mined_dblp_apc_{a_sub}_{p_sub}_{c_sub}",
                nodes_spec=nodes_spec,
                motif_edges=motif_edges,
                support=int(support),
                template="dblp_apc_backchase",
            )
        )

    for (a1_sub, t_sub, a2_sub), support in _top(aat_support):
        nodes_spec = {
            "A1": {"in": [int(a1_sub)]},
            "P1": {"in": paper_allowed},
            "T": {"in": [int(t_sub)]},
            "P2": {"in": paper_allowed},
            "A2": {"in": [int(a2_sub)]},
        }
        motif_edges = [("A1", "P1", 0), ("P1", "T", 1), ("P2", "T", 1), ("A2", "P2", 0)]
        constraints.extend(
            _emit_dblp_backchase_variants(
                prefix=f"mined_dblp_aat_{a1_sub}_{t_sub}_{a2_sub}",
                nodes_spec=nodes_spec,
                motif_edges=motif_edges,
                support=int(support),
                template="dblp_coauthor_topic_backchase",
            )
        )

    for (a1_sub, c_sub, a2_sub), support in _top(aac_support):
        nodes_spec = {
            "A1": {"in": [int(a1_sub)]},
            "P1": {"in": paper_allowed},
            "C": {"in": [int(c_sub)]},
            "P2": {"in": paper_allowed},
            "A2": {"in": [int(a2_sub)]},
        }
        motif_edges = [("A1", "P1", 0), ("P1", "C", 2), ("P2", "C", 2), ("A2", "P2", 0)]
        constraints.extend(
            _emit_dblp_backchase_variants(
                prefix=f"mined_dblp_aac_{a1_sub}_{c_sub}_{a2_sub}",
                nodes_spec=nodes_spec,
                motif_edges=motif_edges,
                support=int(support),
                template="dblp_coauthor_conference_backchase",
            )
        )

    constraints = _dedupe_tgds(constraints)
    constraints = _filter_dblp_constraints_by_consequent_matchability(constraints, dataset_resource, config)
    constraints.sort(
        key=lambda tgd: (
            -int(tgd.get("mining", {}).get("target_workload_hit_count", 0)),
            -int(tgd.get("mining", {}).get("support", 0)),
            str(tgd.get("mining", {}).get("template", "")),
            str(tgd.get("name", "")),
        )
    )
    return constraints[:max_patterns]


def _mine_triangle_constraints(samples: Sequence[Data], dataset_key: str, config: Dict[str, Any]) -> List[TGD]:
    wedge_support = Counter()
    closure_support = Counter()

    for sample in samples:
        labels, neighbors, edge_set = _adjacency(sample)
        wedge_seen = set()
        closed_seen = set()
        for center, nbrs in neighbors.items():
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    la, lc, lb = labels[a], labels[center], labels[b]
                    if la <= lb:
                        key = (la, lc, lb)
                    else:
                        key = (lb, lc, la)
                    wedge_seen.add(key)
                    if _canonical_edge(a, b) in edge_set:
                        closed_seen.add(key)
        wedge_support.update(wedge_seen)
        closure_support.update(closed_seen)

    min_support = int(config.get("constraint_min_support", 3))
    min_conf = float(config.get("constraint_min_confidence", 0.6))
    max_patterns = int(config.get("constraint_max_triangle_patterns", max(1, int(config.get("constraint_max_patterns", 8) // 2) or 1)))

    ranked = []
    for key, support in wedge_support.items():
        if support < min_support:
            continue
        confidence = float(closure_support.get(key, 0)) / float(support)
        if confidence < min_conf:
            continue
        ranked.append((support, confidence, key))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))

    constraints: List[TGD] = []
    for support, confidence, (la, lc, lb) in ranked[:max_patterns]:
        tgd = {
            "name": f"mined_{dataset_key.lower()}_triangle_{la}_{lc}_{lb}",
            "consequent": {
                "nodes": {
                    "A": {"in": [int(la)]},
                    "B": {"in": [int(lb)]},
                    "C": {"in": [int(lc)]},
                },
                "edges": [("A", "C"), ("B", "C")],
                "distinct": ["A", "B", "C"],
            },
            "antecedent": {
                "nodes": {
                    "A": {"in": [int(la)]},
                    "B": {"in": [int(lb)]},
                    "C": {"in": [int(lc)]},
                },
                "edges": [("A", "C"), ("B", "C"), ("A", "B")],
                "distinct": ["A", "B", "C"],
            },
            "mining": {
                "template": "triangle_closure",
                "support": int(support),
                "confidence": float(confidence),
            },
        }
        validate_tgd(tgd)
        constraints.append(tgd)
    return constraints


def _mine_extension_constraints(samples: Sequence[Data], dataset_key: str, config: Dict[str, Any]) -> List[TGD]:
    consequent_support = Counter()
    extension_support = Counter()

    for sample in samples:
        labels, neighbors, edge_set = _adjacency(sample)
        edge_types_seen = set()
        extension_seen = set()

        for u, v in edge_set:
            lu, lv = labels[u], labels[v]
            consequent_key = (lu, lv) if lu <= lv else (lv, lu)
            edge_types_seen.add(consequent_key)

        for anchor, nbrs in neighbors.items():
            for other in nbrs:
                for ext in nbrs:
                    if ext == other:
                        continue
                    extension_seen.add((labels[other], labels[anchor], labels[ext]))

        consequent_support.update(edge_types_seen)
        extension_support.update(extension_seen)

    min_support = int(config.get("constraint_min_support", 3))
    min_conf = float(config.get("constraint_min_confidence", 0.6))
    max_patterns = int(config.get("constraint_max_extension_patterns", max(1, int(config.get("constraint_max_patterns", 8) // 2) or 1)))

    ranked = []
    for key, support in extension_support.items():
        if support < min_support:
            continue
        la, lb, lc = key
        consequent_key = (la, lb) if la <= lb else (lb, la)
        consequent_count = max(1, int(consequent_support.get(consequent_key, 0)))
        confidence = float(support) / float(consequent_count)
        if confidence < min_conf:
            continue
        ranked.append((support, confidence, key))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))

    constraints: List[TGD] = []
    for support, confidence, (la, lb, lc) in ranked[:max_patterns]:
        tgd = {
            "name": f"mined_{dataset_key.lower()}_extension_{la}_{lb}_{lc}",
            "consequent": {
                "nodes": {
                    "A": {"in": [int(la)]},
                    "B": {"in": [int(lb)]},
                },
                "edges": [("A", "B")],
                "distinct": ["A", "B"],
            },
            "antecedent": {
                "nodes": {
                    "A": {"in": [int(la)]},
                    "B": {"in": [int(lb)]},
                    "C": {"in": [int(lc)]},
                },
                "edges": [("A", "B"), ("B", "C")],
                "distinct": ["A", "B", "C"],
            },
            "mining": {
                "template": "edge_extension",
                "support": int(support),
                "confidence": float(confidence),
            },
        }
        validate_tgd(tgd)
        constraints.append(tgd)
    return constraints


def mine_constraints(dataset_resource: Any, config: Dict[str, Any]) -> List[TGD]:
    dataset_key = str(config.get("data_name", "dataset")).upper()
    if dataset_key == "DBLP":
        return _mine_dblp_completion_constraints(dataset_resource, config)
    samples = list(_iter_training_samples(dataset_resource, config))
    if not samples:
        return []

    templates = {str(t).lower() for t in config.get("constraint_templates", ["triangle", "extension"])}
    constraints: List[TGD] = []
    if "triangle" in templates:
        constraints.extend(_mine_triangle_constraints(samples, dataset_key, config))
    if "extension" in templates:
        constraints.extend(_mine_extension_constraints(samples, dataset_key, config))

    max_patterns = int(config.get("constraint_max_patterns", 8))
    constraints.sort(
        key=lambda tgd: (
            -int(tgd.get("mining", {}).get("support", 0)),
            -float(tgd.get("mining", {}).get("confidence", 0.0)),
            str(tgd.get("name", "")),
        )
    )
    return constraints[:max_patterns]


def _write_constraint_bundle(path: str, dataset_key: str, source: str, type_source: str, constraints: Sequence[TGD], config: Dict[str, Any]) -> None:
    bundle = {
        "dataset": dataset_key,
        "constraint_source": source,
        "constraint_type_source": type_source,
        "constraint_rule_mode": str(config.get("constraint_rule_mode", "legacy")).lower(),
        "random_seed": int(config.get("random_seed", 0)),
        "num_constraints": len(constraints),
        "constraints": list(constraints),
        "mining": {
            "constraint_templates": list(config.get("constraint_templates", ["triangle", "extension"])),
            "constraint_min_support": int(config.get("constraint_min_support", 3)),
            "constraint_min_confidence": float(config.get("constraint_min_confidence", 0.6)),
            "constraint_max_patterns": int(config.get("constraint_max_patterns", 8)),
            "constraint_mining_hops": int(config.get("constraint_mining_hops", config.get("L", 2))),
            "constraint_mining_max_samples": int(config.get("constraint_mining_max_samples", 64)),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)


def resolve_constraints(
    config: Dict[str, Any],
    dataset_resource: Any,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> List[TGD]:
    dataset_key = str(config.get("data_name", "dataset")).upper()
    source = str(config.get("constraint_source", "static")).lower()
    type_source = str(config.get("constraint_type_source", "pseudo_type")).lower()

    prepare_constraint_types(config, dataset_resource, model=model, device=device)

    static_constraints = _clone_constraints(get_constraints(dataset_key))
    constraints: List[TGD]

    if source == "static":
        constraints = static_constraints
    else:
        cache_path = _cache_path(config, dataset_key)
        use_cache = bool(config.get("constraint_use_cache", True))
        mined_constraints: List[TGD] = []

        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            mined_constraints = list(bundle.get("constraints", []))
        else:
            mined_constraints = mine_constraints(dataset_resource, config)
            if dataset_key != "DBLP":
                mined_constraints = _filter_constraints_by_consequent_matchability(mined_constraints, dataset_resource, config)
            _write_constraint_bundle(cache_path, dataset_key, source, type_source, mined_constraints, config)

        if source == "hybrid":
            seen = {c.get("name", "") for c in static_constraints}
            constraints = list(static_constraints)
            for tgd in mined_constraints:
                name = tgd.get("name", "")
                if name in seen:
                    continue
                constraints.append(tgd)
                seen.add(name)
        else:
            constraints = mined_constraints

    constraints = _apply_constraint_pool_mode(constraints, config, dataset_key)

    constraint_limit = config.get("constraint_limit", config.get("sigma_size"))
    if constraint_limit is not None:
        try:
            limit = max(0, int(constraint_limit))
            constraints = constraints[:limit]
        except Exception:
            pass

    resolved_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        resolved_path = os.path.join(save_dir, "resolved_constraints.json")
        _write_constraint_bundle(resolved_path, dataset_key, source, type_source, constraints, config)

    config["_resolved_constraints_path"] = resolved_path
    return constraints


__all__ = [
    "mine_constraints",
    "prepare_constraint_types",
    "resolve_constraints",
]
