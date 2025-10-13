

"""Constraint-driven edge masking for triggering backchase on MUTAG (and similar graphs).

设计目标（贴合论文设定）：
- 优先删除 **TGD head** 中的一条边，使得 head 仍可被部分匹配、而 body 需要通过 backchase 修复；
- 避免纯随机删边（随机仅用于在可选候选中打破平 ties）。

依赖：
- matcher.py: 提供 `find_head_matches(data, tgd)`，返回匹配到的变量绑定列表（List[Dict[str,int]]）。
- constraints.py: TGD 结构为 {"name", "head": {nodes, edges, distinct}, "body": {...}}，其中
  nodes: Dict[var_name, {"in": [label_ids]}], edges: List[Tuple[str,str]]，distinct: List[str]

本文件提供：
- mask_edges_by_constraints(data, constraints, max_masks=1, seed=None):
    给定一个 PyG Data（单图）、TGD 列表，优先从 head 匹配里的边中挑选、并在 edge_index 中删除对应无向边（两条反向有向边）。
    返回 (new_data, dropped_edges)，其中 dropped_edges 为 [(u,v), ...] 无向端点对。

注意：
- MUTAG 是无向图，但在 PyG 中通常用双向有向边表示。我们按“无向对”来去重与删除。
- 如果某些 TGD 没有 head 匹配，函数会回退为空操作（不删边）。
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set
import random
import torch
from torch_geometric.data import Data

# 依赖我们自己实现的匹配器
try:
    from src.matcher import find_head_matches  # type: ignore
except Exception:  # 允许相对导入
    from matcher import find_head_matches  # type: ignore

# 依赖常量（可选，不强耦合）：
try:
    from src.constraints import TGD  # Typed alias（如果在 constraints.py 定义了）
except Exception:  # 兼容未定义 Typed alias 的情况
    TGD = Dict[str, Any]


# ------------------------------
# 工具函数
# ------------------------------

def _as_undirected(u: int, v: int) -> Tuple[int, int]:
    """把边端点规范化为无向对 (min,max)。"""
    return (u, v) if u <= v else (v, u)


def _build_edge_bucket(edge_index: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
    """为快速删除构建映射：无向对 -> 在 edge_index 中的列索引列表。
    假设 edge_index 形状 [2, E]。
    """
    bucket: Dict[Tuple[int, int], List[int]] = {}
    src, dst = edge_index[0], edge_index[1]
    for eid in range(edge_index.size(1)):
        key = _as_undirected(int(src[eid]), int(dst[eid]))
        bucket.setdefault(key, []).append(eid)
    return bucket


def _drop_edges(edge_index: torch.Tensor, drop_keys: Set[Tuple[int, int]]) -> torch.Tensor:
    """根据无向键集合 `drop_keys` 从 edge_index 中移除对应列（两向）。"""
    if len(drop_keys) == 0:
        return edge_index
    bucket = _build_edge_bucket(edge_index)
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    for key in drop_keys:
        for eid in bucket.get(key, []):
            keep_mask[eid] = False
    return edge_index[:, keep_mask]


# ------------------------------
# 主功能：基于约束（TGD）进行掩蔽
# ------------------------------

def mask_edges_by_constraints(
    data: Data,
    constraints: List[TGD],
    max_masks: int = 1,
    seed: int | None = None,
    prefer_longer_heads: bool = True,
) -> Tuple[Data, List[Tuple[int, int]]]:
    """优先从各 TGD 的 head 匹配中，选取若干无向边进行删除，以便触发 backchase。

    参数：
        data: PyG 的 Data（单图）。要求包含 x, edge_index（无向图通常用双向有向边表示）。
        constraints: TGD 列表（来自 constraints.py）。
        max_masks: 最多删除多少条“无向边”（默认 1）。
        seed: 随机种子（用于在候选中打破平局），None 表示不固定。
        prefer_longer_heads: 若为 True，则优先从 head 较长（边数更多）的约束里挑边（更容易形成“部分可见”→“需要修复”的情形）。

    返回：
        (new_data, dropped_edges)
        new_data: 拷贝后的 Data，其中 edge_index 已删除对应边。
        dropped_edges: 被删除的无向端点列表 [(u,v), ...]（u<v）。
    """
    if seed is not None:
        random.seed(seed)

    # 收集所有候选 head 边（经由匹配得到变量绑定，再把变量名映射到实际节点 ID，最后把 head.edges 投影为实际边）。
    candidate_keys: List[Tuple[int, int]] = []  # 无向边键集合（带重复，后续去重与打分）
    weighted_pool: List[Tuple[Tuple[int, int], int]] = []  # (无向对, 权重/优先级)

    for tgd in constraints:
        head = tgd.get("head", {})
        head_edges = head.get("edges", [])
        if not head_edges:
            continue

        # 计算优先级：边数越多优先级越高（可调）
        priority = len(head_edges) if prefer_longer_heads else 1

        # 通过匹配器拿到所有 head 的变量绑定 match（dict: var -> node_id）
        try:
            matches = find_head_matches(data, tgd)
        except Exception:
            matches = []

        for bind in matches:
            for (va, vb) in head_edges:
                if va not in bind or vb not in bind:
                    continue
                u, v = int(bind[va]), int(bind[vb])
                key = _as_undirected(u, v)
                candidate_keys.append(key)
                weighted_pool.append((key, priority))

    if not candidate_keys:
        # 没匹配到任何 head，直接返回原图
        return data, []

    # 去重，并根据权重进行一个简单的加权抽样/排序优先选择
    # 策略：按 (priority, 随机噪声) 排序，选前 max_masks 个
    uniq: Dict[Tuple[int, int], int] = {}
    for key, pr in weighted_pool:
        # 记录该 key 的最大 priority（同一无向边可能来自多个 TGD）
        if key not in uniq or pr > uniq[key]:
            uniq[key] = pr

    ranked = sorted(uniq.items(), key=lambda kv: (kv[1], random.random()), reverse=True)
    to_drop: List[Tuple[int, int]] = [kv[0] for kv in ranked[:max_masks]]

    # 实际从 edge_index 删除对应的无向边（两向）
    new_edge_index = _drop_edges(data.edge_index, set(to_drop))

    # 返回新的 Data（浅拷贝 x/batch 等，深拷 edge_index）
    new_data = Data(x=data.x, edge_index=new_edge_index)
    for attr in ("y", "batch"):  # 复制常用字段（若存在）
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))

    return new_data, to_drop


def mask_edges_for_node_classification(
    data: Data,
    target_node: int,
    constraints: List[TGD],
    num_hops: int = 2,
    max_masks: int = 1,
    seed: int | None = None,
    prefer_longer_heads: bool = True,
) -> Tuple[Data, List[Tuple[int, int]], torch.Tensor]:
    """
    Extract L-hop subgraph around target node and apply constraint-based edge masking.
    
    Args:
        data: Full graph Data (node classification)
        target_node: Node index to explain
        constraints: TGD list
        num_hops: L-hop neighborhood size
        max_masks: Max edges to remove
        seed: Random seed
        prefer_longer_heads: Prioritize longer HEAD patterns
        
    Returns:
        (masked_subgraph, dropped_edges, node_subset)
        masked_subgraph: L-hop subgraph with edges masked
        dropped_edges: List of dropped edge pairs (in subgraph IDs)
        node_subset: Original node IDs in the subgraph
    """
    from torch_geometric.utils import k_hop_subgraph
    
    # Step 1: Extract L-hop subgraph
    node_subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    
    # Step 2: Build subgraph Data object
    x_sub = data.x[node_subset]
    y_sub = data.y[node_subset] if hasattr(data, 'y') and data.y is not None else None
    
    # Find target node's new ID in the subgraph
    # node_subset contains original node IDs, we need to find where target_node appears
    target_node_subgraph_id = (node_subset == target_node).nonzero(as_tuple=True)[0].item()
    
    subgraph = Data(x=x_sub, edge_index=edge_index_sub)
    if y_sub is not None:
        subgraph.y = y_sub
    subgraph.num_nodes = int(node_subset.numel())
    subgraph.original_node_ids = node_subset
    subgraph.target_node_subgraph_id = target_node_subgraph_id
    subgraph.target_node_original_id = target_node
    
    # Step 3: Apply constraint-based masking on the subgraph
    masked_subgraph, dropped_edges = mask_edges_by_constraints(
        subgraph,
        constraints,
        max_masks=max_masks,
        seed=seed,
        prefer_longer_heads=prefer_longer_heads,
    )
    
    # Copy over metadata
    masked_subgraph.original_node_ids = node_subset
    masked_subgraph.target_node_subgraph_id = subgraph.target_node_subgraph_id
    masked_subgraph.target_node_original_id = target_node
    
    return masked_subgraph, dropped_edges, node_subset


__all__ = [
    "mask_edges_by_constraints",
    "mask_edges_for_node_classification",
]