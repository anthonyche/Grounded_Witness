
from typing import Dict, List, Tuple, Any, Set, Iterable, Optional

# 说明：
#  - 本文件提供约束(TGD)匹配与“回追(backchase)”的最小实现。
#  - 充分利用 NetworkX 的 VF2 同构算法来做模式匹配（head 模式）。
#  - 为了简单、实用：假定节点特征为 one-hot，类别为 argmax(x)；边无类型；图按无向处理。
#  - repair cost 的估计采用启发式（greedy）：当 body 中出现 head 未绑定的新变量时，
#    尝试在图中用满足类型约束且连边存在的真实节点来“临时绑定”；若找不到，则计为缺边 1。
#  - 以上实现够用即可，不追求最优/完整搜索；后续如需更强大，可再迭代。

import networkx as nx
import torch

# ------------------------------
# 小工具：从 PyG 子图/数据结构 构造成 NetworkX 图
# ------------------------------

def _data_to_nx(Gs: Any) -> nx.Graph:
    """
    将传入的子图 Gs 归一化为无向 NetworkX 图，
    并在每个节点属性中写入：
      - label: 来自 one-hot 节点特征的 argmax，或使用预设的 node_labels
    支持两种最常见输入：
      1) PyG Data 对象（含 x: Tensor, edge_index: Tensor[2, E]）
      2) 轻量结构：具有 .x (Tensor[N, F]) 和 .edges (List[Tuple[int,int]])
    """
    G = nx.Graph()

    # 处理节点与标签
    # First, check if we have explicit node_labels attribute (for datasets without discrete labels)
    if hasattr(Gs, 'node_labels') and isinstance(Gs.node_labels, torch.Tensor):
        labels = Gs.node_labels.tolist()
        for i, lab in enumerate(labels):
            G.add_node(int(i), label=int(lab))
    elif hasattr(Gs, 'x') and isinstance(Gs.x, torch.Tensor):
        x = Gs.x
        # For datasets with discrete labels (one-hot), use argmax
        # For feature-based datasets (like Yelp), assign dummy label 0 (all nodes same type)
        if x.size(1) <= 10:  # Heuristic: likely one-hot encoding
            labels = x.argmax(dim=-1).tolist()
        else:  # Likely continuous features - assign dummy label
            labels = [0] * x.size(0)
        for i, lab in enumerate(labels):
            G.add_node(int(i), label=int(lab))
    else:
        raise ValueError("Gs 需要包含张量属性 x 以便读取节点标签(one-hot)")

    # 处理边
    if hasattr(Gs, 'edge_index') and isinstance(Gs.edge_index, torch.Tensor):
        ei = Gs.edge_index
        assert ei.dim() == 2 and ei.size(0) == 2, "edge_index 形状需为 [2, E]"
        u = ei[0].tolist(); v = ei[1].tolist()
        for a, b in zip(u, v):
            if a == b:
                continue
            G.add_edge(int(a), int(b))
    elif hasattr(Gs, 'edges') and isinstance(Gs.edges, list):
        for a, b in Gs.edges:
            if a == b:
                continue
            G.add_edge(int(a), int(b))
    else:
        raise ValueError("Gs 需要包含 edge_index(Tensor) 或 edges(List[Tuple[int,int]])")

    return G

# ------------------------------
# 从 TGD 的 head/body 规范 构造“模式图”
# ------------------------------

def _build_pattern(spec: Dict[str, Any]) -> nx.Graph:
    """
    根据 TGD 里 head/body 的字典规范构造一个模式图（无向）。
    节点属性：
      - allowed: Set[int]，允许的标签集合（例如 {C} 或 {C,N}）
    边：元组("U","V")，使用变量名作为节点 id。
    """
    P = nx.Graph()
    nodes: Dict[str, Dict[str, Any]] = spec.get('nodes', {})
    edges: Iterable[Tuple[str, str]] = spec.get('edges', [])

    for var, cond in nodes.items():
        allowed = set(cond.get('in', []))
        P.add_node(var, allowed=allowed)
    for a, b in edges:
        if a == b:
            continue
        P.add_edge(a, b)
    return P

# 自定义的节点匹配函数：
# 模式节点有 allowed 集合；目标图节点有单一 label；需要 label ∈ allowed。

def _node_match(attrs_g: Dict[str, Any], attrs_p: Dict[str, Any]) -> bool:
    label = attrs_g.get('label', None)
    allowed = attrs_p.get('allowed', None)
    if allowed is None:
        return True
    if label is None:
        return False
    return (label in allowed)

# ------------------------------
# 对外 API：查找 head 匹配
# ------------------------------

def find_head_matches(Gs: Any, tgd: Dict[str, Any]) -> List[Dict[str, int]]:
    """
    使用 VF2 在 Gs 中查找 tgd['head'] 的所有匹配。
    返回：每个匹配为 dict[var_name -> node_id]
    说明：VF2 默认是一一映射，已隐含满足 distinct 约束；如需更多自定义可扩展。
    """
    G = _data_to_nx(Gs)
    head = tgd['head']
    P = _build_pattern(head)

    GM = nx.algorithms.isomorphism.GraphMatcher(
        G, P,
        node_match=lambda ng, np: _node_match(ng, np)
    )

    results: List[Dict[str, int]] = []
    for mapping in GM.subgraph_isomorphisms_iter():
        # mapping: target_node_id -> pattern_var
        inv: Dict[str, int] = {}
        for n_g, n_p in mapping.items():
            inv[n_p] = int(n_g)
        # 可选：检查 distinct 列表（尽管 VF2 已保证不同变量映到不同节点）
        distinct = head.get('distinct', [])
        if distinct:
            bound_nodes = [inv[v] for v in distinct if v in inv]
            if len(bound_nodes) != len(set(bound_nodes)):
                continue
        results.append(inv)
    return results

# ------------------------------
# 回追(Backchase) + 维修代价估计（启发式）
# ------------------------------

def _edge_exists(G: nx.Graph, u: int, v: int) -> bool:
    return G.has_edge(u, v) or G.has_edge(v, u)


def _neighbors_of_label(G: nx.Graph, u: int, allowed: Set[int]) -> List[int]:
    """在 u 的邻居中，筛选出 label ∈ allowed 的节点。"""
    out = []
    for w in G.neighbors(u):
        lab = G.nodes[w].get('label', None)
        if lab in allowed:
            out.append(w)
    return out


def backchase_repair_cost(Gs: Any, tgd: Dict[str, Any], binding: Dict[str, int], B: int) -> Tuple[bool, int, List[Tuple[int, int]]]:
    """
    给定一个 head 匹配 binding，估计实现 body 所需的(最小)“缺边数”。
    返回：(是否在预算内, 估计代价rep, 需要插入的边列表 repairs)

    启发式策略：按 body.edges 顺序处理，维护一个“扩展的变量绑定 env”。
      - 若边的两个端点变量都已绑定，若图中无此边 -> 缺边+1。
      - 若一端绑定、另一端未绑定：尝试在邻居中找满足类型且连边已存在的节点来绑定；
        若找不到 -> 缺边+1（视作需要插入这一条边）。
      - 若两端都未绑定：尝试直接在图中找到一条满足两端类型的现有边(任意挑一个)；
        若找不到 -> 缺边+1。
    注意：这是近似的下界/上界混合启发，追求快速，不保证全局最优。
    """
    G = _data_to_nx(Gs)
    body = tgd['body']
    spec_nodes: Dict[str, Dict[str, Any]] = body.get('nodes', {})
    body_edges: Iterable[Tuple[str, str]] = body.get('edges', [])

    # 初始把 head 绑定放入 env
    env: Dict[str, int] = dict(binding)
    repairs: List[Tuple[int, int]] = []
    rep = 0

    # 为未绑定的变量准备其允许标签集合
    allowed_map: Dict[str, Set[int]] = {
        var: set(cond.get('in', [])) for var, cond in spec_nodes.items()
    }

    for u_var, v_var in body_edges:
        u_bound = u_var in env
        v_bound = v_var in env

        if u_bound and v_bound:
            u_id, v_id = env[u_var], env[v_var]
            if not _edge_exists(G, u_id, v_id):
                rep += 1
                repairs.append((u_id, v_id))
                if rep > B:
                    return (False, rep, repairs)
            continue

        if u_bound and not v_bound:
            u_id = env[u_var]
            cand = _neighbors_of_label(G, u_id, allowed_map.get(v_var, set()))
            if cand:
                env[v_var] = cand[0]  # 任取一个
            else:
                # 无合适邻居，视作需要插入 (u_id, new_v)
                rep += 1
                repairs.append((u_id, -1))  # -1 表示需要新接入/未知节点
                if rep > B:
                    return (False, rep, repairs)
            continue

        if not u_bound and v_bound:
            v_id = env[v_var]
            cand = _neighbors_of_label(G, v_id, allowed_map.get(u_var, set()))
            if cand:
                env[u_var] = cand[0]
            else:
                rep += 1
                repairs.append((-1, v_id))
                if rep > B:
                    return (False, rep, repairs)
            continue

        # 两端都未绑定：尝试在图里找一条满足标签的现有边
        set_u = allowed_map.get(u_var, set())
        set_v = allowed_map.get(v_var, set())
        found_pair: Optional[Tuple[int, int]] = None
        for a, b in G.edges():
            la = G.nodes[a].get('label', None)
            lb = G.nodes[b].get('label', None)
            if la in set_u and lb in set_v:
                found_pair = (a, b)
                break
            if la in set_v and lb in set_u:
                found_pair = (b, a)
                break
        if found_pair is not None:
            env[u_var], env[v_var] = found_pair
        else:
            rep += 1
            repairs.append((-1, -1))
            if rep > B:
                return (False, rep, repairs)

    return (True, rep, repairs)

# ------------------------------
# 计算 Γ(Gs, B) ：在预算内可“落地”的约束集合（按名字返回）
# ------------------------------

def Gamma(Gs: Any, B: int, tgd_list: List[Dict[str, Any]], max_matches_per_tgd: int = 5) -> Set[str]:
    """
    返回：在子图 Gs 上、在预算 B 内能够“grounded”的约束名称集合。
    做法：
      1) 先用 VF2 找每条 TGD 的 head 匹配列表；
      2) 对每个匹配调用 backchase 估算 repair cost；若 ≤ B，则记为可落地；
      3) 只要某条 TGD 存在一个匹配能落地，即计入 Γ。
    """
    grounded: Set[str] = set()
    for tgd in tgd_list:
        name = tgd.get('name', 'unnamed')
        matches = find_head_matches(Gs, tgd)
        # Limit matches to avoid combinatorial explosion
        if len(matches) > max_matches_per_tgd:
            matches = matches[:max_matches_per_tgd]
        ok = False
        for bind in matches:
            feasible, rep, _ = backchase_repair_cost(Gs, tgd, bind, B)
            if feasible:
                ok = True
                break
        if ok:
            grounded.add(name)
    return grounded

# 方便外部复用的“结果容器”（简化版）
class MatchResult:
    """简单封装：是否落地/代价/建议补边。"""
    def __init__(self, grounded: bool, rep_cost: int, repairs: List[Tuple[int, int]]):
        self.grounded = grounded
        self.rep_cost = rep_cost
        self.repairs = repairs
