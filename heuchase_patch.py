"""
Patch for heuchase._candidate_by_edmonds to fix TreeCycle slowness

Issues fixed:
1. NaN weights causing Edmonds to hang
2. Nearly identical weights causing tie-breaking issues
3. Better fallback when Edmonds fails

Usage:
    from heuchase_patch import candidate_by_edmonds_fixed
    explainer._candidate_by_edmonds = candidate_by_edmonds_fixed
"""

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional
import numpy as np


def _cos_sim_safe(a: Tensor, b: Tensor, eps=1e-8) -> float:
    """Safe cosine similarity with NaN/Inf protection"""
    if a is None or b is None:
        return 0.0
    
    # Normalize with epsilon
    norm_a = max(a.norm(p=2).item(), eps)
    norm_b = max(b.norm(p=2).item(), eps)
    
    dot_product = torch.dot(a, b).item()
    
    # Check for NaN/Inf
    if not np.isfinite(dot_product) or not np.isfinite(norm_a) or not np.isfinite(norm_b):
        return 0.0
    
    cos_val = dot_product / (norm_a * norm_b)
    
    # Clip to valid range and check again
    if not np.isfinite(cos_val):
        return 0.0
    
    return float(np.clip(cos_val, -1.0, 1.0))


def candidate_by_edmonds_fixed(H: Data, root: Optional[int], emb: Optional[Tensor], noise_std: float = 1e-3):
    """
    Fixed version of _candidate_by_edmonds with:
    - Safe cosine similarity (no NaN)
    - Larger noise for tie-breaking
    - Timeout protection
    - Better fallback
    """
    E = H.edge_index.size(1)
    if E == 0:
        return torch.zeros(E, dtype=torch.bool, device=H.edge_index.device)

    device = H.edge_index.device
    edge_mask = torch.zeros(E, dtype=torch.bool, device=device)
    
    # Try Edmonds with timeout protection
    try:
        import networkx as nx
        import signal
        
        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Edmonds timeout")
        
        # Set 2-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        
        try:
            G = nx.DiGraph()
            N = int(H.num_nodes if getattr(H, 'num_nodes', None) is not None else H.x.size(0))
            
            for n in range(N):
                G.add_node(n)
            
            # Compute weights with safety checks
            rng = torch.Generator()
            rng.manual_seed(torch.randint(0, 10_000_000, (1,)).item())
            
            # Larger noise for better tie-breaking (0.01 instead of 0.001)
            noise_scale = max(noise_std, 0.01)
            eps = torch.randn(E, generator=rng).tolist()
            
            weights_computed = 0
            nan_count = 0
            
            for idx in range(E):
                u = int(H.edge_index[0, idx].item())
                v = int(H.edge_index[1, idx].item())
                
                # Safe cosine similarity
                w_uv = _cos_sim_safe(emb[u], emb[v]) if emb is not None else 0.0
                w_vu = _cos_sim_safe(emb[v], emb[u]) if emb is not None else 0.0
                
                # Add unique noise per direction
                nse_uv = float(eps[idx]) * noise_scale
                nse_vu = float(eps[(idx + 1) % E]) * noise_scale  # Different noise for reverse
                
                # Add hash-based micro-perturbation for uniqueness
                hash_uv = hash((u, v, 'forward')) % 10000 / 1e6
                hash_vu = hash((v, u, 'backward')) % 10000 / 1e6
                
                final_w_uv = w_uv + nse_uv + hash_uv
                final_w_vu = w_vu + nse_vu + hash_vu
                
                # Check for NaN
                if not np.isfinite(final_w_uv):
                    nan_count += 1
                    final_w_uv = 0.01 * (idx / E)  # Fallback: small increasing weights
                if not np.isfinite(final_w_vu):
                    nan_count += 1
                    final_w_vu = 0.01 * ((idx + 0.5) / E)
                
                # Add edges with unique IDs
                G.add_edge(u, v, weight=final_w_uv, _eid=idx, _dir='forward')
                G.add_edge(v, u, weight=final_w_vu, _eid=idx, _dir='backward')
                
                weights_computed += 2
            
            if nan_count > 0:
                print(f"Warning: {nan_count}/{weights_computed} weights were NaN (fixed)", flush=True)
            
            # Try Edmonds
            Ar = nx.maximum_spanning_arborescence(G, attr='weight', default=0)
            
            # Build mapping
            edge_to_eid = {}
            for u, v, data in G.edges(data=True):
                edge_to_eid[(u, v)] = data.get('_eid', None)
            
            # Find root
            if root is None:
                ar_roots = [n for n in Ar.nodes if Ar.in_degree(n) == 0]
                root = ar_roots[0] if ar_roots else (max(Ar.nodes, key=lambda n: Ar.out_degree(n)) if Ar.number_of_nodes() > 0 else 0)
            
            # BFS from root
            if root in Ar:
                from collections import deque
                dq = deque([root])
                seen = set([root])
                while dq:
                    u = dq.popleft()
                    for _, v in Ar.out_edges(u):
                        eid = edge_to_eid.get((u, v), None)
                        if eid is not None:
                            edge_mask[eid] = True
                        if v not in seen:
                            seen.add(v)
                            dq.append(v)
            
            # Success! Cancel alarm
            signal.alarm(0)
            
            if edge_mask.sum().item() > 0:
                return edge_mask
        
        except TimeoutError:
            signal.alarm(0)
            print("Warning: Edmonds timed out after 2s, using fallback", flush=True)
        except Exception as e:
            signal.alarm(0)
            print(f"Warning: Edmonds failed ({str(e)[:50]}), using fallback", flush=True)
    
    except ImportError:
        print("Warning: NetworkX not available, using fallback", flush=True)
    
    # Fallback: Greedy MST-like selection
    print("Using greedy fallback (MST-like)", flush=True)
    
    # Use degree-based heuristic + some randomness
    if root is not None:
        # Star from root
        r = int(root)
        for idx in range(E):
            u = int(H.edge_index[0, idx].item())
            v = int(H.edge_index[1, idx].item())
            if u == r or v == r:
                edge_mask[idx] = True
        
        # Add more edges greedily
        seen = set([r])
        for idx in range(E):
            if edge_mask[idx]:
                u = int(H.edge_index[0, idx].item())
                v = int(H.edge_index[1, idx].item())
                seen.add(u)
                seen.add(v)
        
        for idx in range(E):
            if edge_mask.sum().item() >= N - 1:  # Tree has N-1 edges
                break
            u = int(H.edge_index[0, idx].item())
            v = int(H.edge_index[1, idx].item())
            if (u in seen) != (v in seen):  # One endpoint in tree, one not
                edge_mask[idx] = True
                seen.add(u)
                seen.add(v)
    else:
        # No root: just take first N-1 edges
        edge_mask[:min(N-1, E)] = True
    
    return edge_mask
