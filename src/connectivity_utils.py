"""Utility functions for checking graph connectivity during masking."""

from typing import Tuple, Set, List
import torch


def is_bridge_edge(edge_index: torch.Tensor, num_nodes: int, edge_key: Tuple[int, int]) -> bool:
    """Check if removing edge_key would disconnect the graph (i.e., if it's a bridge edge).
    
    Uses BFS to check:
    1. Temporarily remove the edge
    2. BFS from one endpoint to see if we can reach the other
    3. If unreachable, it's a bridge
    
    Args:
        edge_index: PyG edge_index tensor [2, E]
        num_nodes: Total number of nodes
        edge_key: Undirected edge (u, v) where u < v
        
    Returns:
        True if edge is a bridge (removing would disconnect graph)
    """
    try:
        import networkx as nx
    except ImportError:
        # Conservative: assume NOT a bridge if networkx unavailable (allow removal)
        return False
    
    # Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Add all edges
    src, dst = edge_index[0], edge_index[1]
    for i in range(edge_index.size(1)):
        u, v = int(src[i]), int(dst[i])
        if u < v:  # Only add once (undirected)
            G.add_edge(u, v)
    
    # Check if original graph is connected
    if not nx.is_connected(G):
        # Already disconnected, removing any edge won't "break" connectivity
        return False
    
    # Temporarily remove target edge
    u, v = edge_key
    if not G.has_edge(u, v):
        return False  # Edge doesn't exist, not a bridge
    
    G.remove_edge(u, v)
    
    # Check if still connected after removal
    is_still_connected = nx.is_connected(G)
    
    return not is_still_connected  # Bridge if removal disconnects graph


def filter_non_bridge_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    candidate_edges: Set[Tuple[int, int]],
    verbose: bool = True
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Filter candidate edges to separate bridge edges from non-bridge edges.
    
    Args:
        edge_index: PyG edge_index tensor [2, E]
        num_nodes: Total number of nodes
        candidate_edges: Set of (u, v) edge keys to check
        verbose: Print debug information
        
    Returns:
        (non_bridge_edges, bridge_edges) - two disjoint sets
    """
    non_bridge = set()
    bridges = set()
    
    for edge in candidate_edges:
        if is_bridge_edge(edge_index, num_nodes, edge):
            bridges.add(edge)
        else:
            non_bridge.add(edge)
    
    if verbose and bridges:
        print(f"[CONNECTIVITY] Found {len(bridges)} bridge edges out of {len(candidate_edges)} candidates")
        print(f"[CONNECTIVITY] Bridge edges: {list(bridges)[:3]}{'...' if len(bridges) > 3 else ''}")
        print(f"[CONNECTIVITY] Safe to remove: {len(non_bridge)} non-bridge edges")
    
    return non_bridge, bridges


def select_edges_preserving_connectivity(
    edge_index: torch.Tensor,
    num_nodes: int,
    ranked_candidates: List[Tuple[Tuple[int, int], int]],
    max_masks: int,
    verbose: bool = True
) -> List[Tuple[int, int]]:
    """Incrementally select edges to remove while preserving graph connectivity.
    
    This function selects edges one by one, checking after each selection
    whether removing all selected edges would keep the graph connected.
    
    Args:
        edge_index: PyG edge_index tensor [2, E]
        num_nodes: Total number of nodes
        ranked_candidates: List of (edge_key, priority) sorted by priority
        max_masks: Maximum number of edges to remove
        verbose: Print debug information
        
    Returns:
        List of edge keys to remove
    """
    try:
        import networkx as nx
    except ImportError:
        if verbose:
            print("[CONNECTIVITY WARN] NetworkX not available, falling back to simple selection")
        return [edge for edge, _ in ranked_candidates[:max_masks]]
    
    selected_edges = []
    
    for edge_key, priority in ranked_candidates:
        if len(selected_edges) >= max_masks:
            break
        
        # Build a temporary graph with all selected edges removed
        test_removed = set(selected_edges + [edge_key])
        
        # Build NetworkX graph
        G_test = nx.Graph()
        G_test.add_nodes_from(range(num_nodes))
        src, dst = edge_index[0], edge_index[1]
        for i in range(edge_index.size(1)):
            u, v = int(src[i]), int(dst[i])
            if u < v:  # Only add once (undirected)
                edge = (u, v)
                if edge not in test_removed:
                    G_test.add_edge(u, v)
        
        # Check if graph remains connected
        if nx.is_connected(G_test):
            selected_edges.append(edge_key)
            if verbose:
                print(f"[CONNECTIVITY] Selected edge {edge_key} (priority={priority}), total={len(selected_edges)}/{max_masks}")
        else:
            if verbose:
                print(f"[CONNECTIVITY] Skipped edge {edge_key} - would disconnect graph")
    
    if verbose and len(selected_edges) < max_masks:
        print(f"[CONNECTIVITY] Only {len(selected_edges)}/{max_masks} edges can be safely removed")
    
    return selected_edges


__all__ = ["is_bridge_edge", "filter_non_bridge_edges", "select_edges_preserving_connectivity"]
