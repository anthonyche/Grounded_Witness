"""
新版本的repair cost计算逻辑 (2025-10-13)

修改原因: repair cost应该定义为clean graph中BODY需要的边与witness中已有边的差值
"""

from typing import Dict, List, Tuple, Any, Set, Optional
import networkx as nx


def _data_to_nx_simple(Gs: Any) -> nx.Graph:
    """简化版本的数据转换"""
    if isinstance(Gs, nx.Graph):
        return Gs
    
    import torch
    G = nx.Graph()
    
    # 添加节点
    num_nodes = Gs.num_nodes if hasattr(Gs, 'num_nodes') else Gs.x.size(0)
    for i in range(num_nodes):
        G.add_node(i)
        if hasattr(Gs, 'x') and Gs.x is not None:
            feat = Gs.x[i]
            if torch.is_tensor(feat):
                label = int(torch.argmax(feat).item())
                G.nodes[i]['label'] = label
    
    # 添加边
    if hasattr(Gs, 'edge_index'):
        edge_index = Gs.edge_index
        for j in range(edge_index.size(1)):
            u, v = int(edge_index[0, j]), int(edge_index[1, j])
            G.add_edge(u, v)
    
    return G


def _edge_exists_simple(G: nx.Graph, u: int, v: int) -> bool:
    """检查边是否存在(无向)"""
    return G.has_edge(u, v) or G.has_edge(v, u)


def _neighbors_of_label_simple(G: nx.Graph, u: int, allowed: Set[int]) -> List[int]:
    """返回u的邻居中标签在allowed中的节点"""
    out = []
    for w in G.neighbors(u):
        lab = G.nodes[w].get('label', None)
        if lab in allowed:
            out.append(w)
    return out


def backchase_repair_cost_v2(Gs_clean: Any, tgd: Dict[str, Any], binding: Dict[str, int], B: int, 
                               witness_nodes: Optional[Set[int]] = None, 
                               witness_edges: Optional[Set[Tuple[int, int]]] = None) -> Tuple[bool, int, List[Tuple[int, int]]]:
    """
    新版本的repair cost计算 (2025-10-13)
    
    逻辑:
    1. 在clean graph上尝试完成BODY的绑定(不受witness限制)
    2. 如果成功,收集所有BODY需要的边
    3. repair cost = BODY边中不在witness中的边数
    4. 返回这些需要repair的边
    
    参数:
      Gs_clean: clean graph (完整的原始图)
      tgd: TGD约束定义
      binding: HEAD匹配的变量绑定
      B: 预算上限
      witness_nodes: witness包含的节点集合 (可选,用于验证)
      witness_edges: witness包含的边集合 (无向,应包含(u,v)和(v,u))
    
    返回:
      (是否在预算内, repair代价, 需要repair的边列表)
    """
    G = _data_to_nx_simple(Gs_clean)
    body = tgd['body']
    spec_nodes: Dict[str, Dict[str, Any]] = body.get('nodes', {})
    body_edges: List[Tuple[str, str]] = list(body.get('edges', []))
    distinct_vars = body.get('distinct', [])
    
    # Debug flag
    import os
    debug = os.environ.get('DEBUG_MATCHER_V2', '') == '1'
    
    if debug:
        print(f"\n[MATCHER_V2] Starting repair cost calculation")
        print(f"[MATCHER_V2]   binding={binding}")
        print(f"[MATCHER_V2]   witness_nodes={sorted(witness_nodes) if witness_nodes else 'None'}")
        print(f"[MATCHER_V2]   witness_edges_count={len(witness_edges) if witness_edges else 0}")
        print(f"[MATCHER_V2]   body_edges={body_edges}")
        print(f"[MATCHER_V2]   distinct_vars={distinct_vars}")
    
    # 第一步: 在clean graph上尝试完成BODY绑定
    env: Dict[str, int] = dict(binding)
    allowed_map: Dict[str, Set[int]] = {
        var: set(cond.get('in', [])) for var, cond in spec_nodes.items()
    }
    
    # 尝试绑定所有BODY变量
    for u_var, v_var in body_edges:
        u_bound = u_var in env
        v_bound = v_var in env
        
        if debug:
            print(f"[MATCHER_V2]   Processing edge ({u_var}, {v_var}): u_bound={u_bound}, v_bound={v_bound}")
        
        if u_bound and v_bound:
            # 两端都已绑定,检查边是否存在
            u_id, v_id = env[u_var], env[v_var]
            if not _edge_exists_simple(G, u_id, v_id):
                if debug:
                    print(f"[MATCHER_V2]     Edge ({u_id}, {v_id}) MISSING in clean graph - FAIL")
                return (False, B + 1, [])
            if debug:
                print(f"[MATCHER_V2]     Edge ({u_id}, {v_id}) EXISTS in clean graph")
            continue
        
        if u_bound and not v_bound:
            u_id = env[u_var]
            cand = _neighbors_of_label_simple(G, u_id, allowed_map.get(v_var, set()))
            if debug:
                print(f"[MATCHER_V2]     Finding {v_var} neighbors of {u_id}: candidates={cand}")
            # 检查distinct约束
            if distinct_vars and v_var in distinct_vars:
                already_bound = set(env[dv] for dv in distinct_vars if dv in env and dv != v_var)
                cand = [n for n in cand if n not in already_bound]
                if debug:
                    print(f"[MATCHER_V2]     After distinct filter: candidates={cand}")
            if cand:
                env[v_var] = cand[0]
                if debug:
                    print(f"[MATCHER_V2]     Bound {v_var}={cand[0]}")
            else:
                if debug:
                    print(f"[MATCHER_V2]     No candidate found for {v_var} - FAIL")
                return (False, B + 1, [])
            continue
        
        if not u_bound and v_bound:
            v_id = env[v_var]
            cand = _neighbors_of_label_simple(G, v_id, allowed_map.get(u_var, set()))
            # 检查distinct约束
            if distinct_vars and u_var in distinct_vars:
                already_bound = set(env[dv] for dv in distinct_vars if dv in env and dv != u_var)
                cand = [n for n in cand if n not in already_bound]
            if cand:
                env[u_var] = cand[0]
                if debug:
                    print(f"[MATCHER_V2]     Bound {u_var}={cand[0]}")
            else:
                if debug:
                    print(f"[MATCHER_V2]     No candidate found for {u_var} - FAIL")
                return (False, B + 1, [])
            continue
        
        # 两端都未绑定
        set_u = allowed_map.get(u_var, set())
        set_v = allowed_map.get(v_var, set())
        already_bound = set()
        if distinct_vars:
            if u_var in distinct_vars or v_var in distinct_vars:
                already_bound = set(env[dv] for dv in distinct_vars if dv in env)
        found_pair: Optional[Tuple[int, int]] = None
        for a, b in G.edges():
            if already_bound and (a in already_bound or b in already_bound):
                continue
            la = G.nodes[a].get('label', None)
            lb = G.nodes[b].get('label', None)
            if la in set_u and lb in set_v:
                found_pair = (a, b)
                break
            if la in set_v and lb in set_u:
                found_pair = (b, a)
                break
        if found_pair:
            env[u_var] = found_pair[0]
            env[v_var] = found_pair[1]
            if debug:
                print(f"[MATCHER_V2]     Bound {u_var}={found_pair[0]}, {v_var}={found_pair[1]}")
        else:
            if debug:
                print(f"[MATCHER_V2]     No candidate pair found for ({u_var}, {v_var}) - FAIL")
            return (False, B + 1, [])
    
    if debug:
        print(f"[MATCHER_V2]   Final binding: {env}")
    
    # 第二步: BODY在clean graph上可以满足,收集所有BODY边
    body_concrete_edges: List[Tuple[int, int]] = []
    for u_var, v_var in body_edges:
        if u_var in env and v_var in env:
            u_id, v_id = env[u_var], env[v_var]
            body_concrete_edges.append((u_id, v_id))
    
    if debug:
        print(f"[MATCHER_V2]   BODY concrete edges: {body_concrete_edges}")
    
    # 第三步: 计算哪些BODY边不在witness中
    repairs: List[Tuple[int, int]] = []
    if witness_edges is not None:
        for u, v in body_concrete_edges:
            # 检查无向边
            if (u, v) not in witness_edges and (v, u) not in witness_edges:
                repairs.append((u, v))
                if debug:
                    print(f"[MATCHER_V2]     Edge ({u}, {v}) NOT in witness - needs repair")
            else:
                if debug:
                    print(f"[MATCHER_V2]     Edge ({u}, {v}) already in witness")
    else:
        # 如果没有提供witness_edges,则所有BODY边都算作repair
        repairs = body_concrete_edges
        if debug:
            print(f"[MATCHER_V2]   No witness_edges provided, all BODY edges count as repairs")
    
    rep_cost = len(repairs)
    within_budget = rep_cost <= B
    
    if debug:
        print(f"[MATCHER_V2]   Final: rep_cost={rep_cost}, within_budget={within_budget}, repairs={repairs}\n")
    
    return (within_budget, rep_cost, repairs)
