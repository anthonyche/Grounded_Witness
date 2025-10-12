"""
node_explainer_utils.py
-----------------------
Utilities for node-level explanation tasks, including:
  - L-hop subgraph extraction around target nodes
  - Node classification specific operations
  - Adaptation layer between node classification and existing graph-level code
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from typing import List, Tuple, Set, Optional
import networkx as nx


def extract_l_hop_subgraph(
    data: Data,
    target_node: int,
    num_hops: int,
    relabel_nodes: bool = True
) -> Tuple[Data, torch.Tensor, int]:
    """
    Extract L-hop subgraph around a target node for node classification explanation.
    
    Args:
        data: Full graph data (PyG Data object)
        target_node: Node index to explain
        num_hops: Number of hops (L) for neighborhood extraction
        relabel_nodes: If True, relabel nodes to 0-based consecutive IDs
        
    Returns:
        subgraph_data: PyG Data object for the L-hop subgraph
        node_mapping: Tensor mapping new node IDs to original node IDs
        target_node_new: New ID of target node in subgraph (0 if relabel_nodes=True)
    """
    # Use PyG's k_hop_subgraph to get subset of nodes and edges
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=data.num_nodes,
    )
    
    # Extract node features for the subgraph
    x_sub = data.x[subset]
    
    # Extract labels if available
    y_sub = None
    if hasattr(data, 'y') and data.y is not None:
        if data.y.dim() == 1:  # Single-label
            y_sub = data.y[subset]
        else:  # Multi-label
            y_sub = data.y[subset]
    
    # Create subgraph Data object
    subgraph_data = Data(
        x=x_sub,
        edge_index=edge_index_sub,
        y=y_sub,
    )
    
    # Add edge attributes if they exist
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        subgraph_data.edge_attr = data.edge_attr[edge_mask]
    
    # Determine target node's new ID
    if relabel_nodes:
        # mapping gives the new ID of the target node
        target_node_new = mapping.item()
    else:
        target_node_new = target_node
    
    # Store original node IDs for reference
    subgraph_data.original_node_ids = subset
    subgraph_data.target_node_subgraph_id = target_node_new
    subgraph_data.target_node_original_id = target_node
    
    return subgraph_data, subset, target_node_new


def subgraph_to_full_node_mapping(
    subset: torch.Tensor,
    num_nodes_full: int
) -> torch.Tensor:
    """
    Create a mapping tensor from subgraph node IDs back to full graph node IDs.
    
    Args:
        subset: Tensor of original node IDs in the subgraph
        num_nodes_full: Total number of nodes in the full graph
        
    Returns:
        mapping: Tensor of shape [num_nodes_subgraph] where mapping[i] = original_node_id
    """
    return subset


def edges_to_full_graph_ids(
    edge_list: List[Tuple[int, int]],
    subset: torch.Tensor
) -> List[Tuple[int, int]]:
    """
    Convert edge list from subgraph IDs to full graph IDs.
    
    Args:
        edge_list: List of (u, v) tuples in subgraph node IDs
        subset: Tensor mapping subgraph IDs to original node IDs
        
    Returns:
        edges_full: List of (u, v) tuples in original node IDs
    """
    subset_list = subset.tolist()
    edges_full = []
    for u, v in edge_list:
        u_orig = subset_list[u]
        v_orig = subset_list[v]
        edges_full.append((u_orig, v_orig))
    return edges_full


def get_subgraph_node_labels(
    data: Data,
    subset: torch.Tensor
) -> torch.Tensor:
    """
    Get node labels for nodes in a subgraph.
    
    Args:
        data: Full graph data
        subset: Tensor of node indices in the subgraph
        
    Returns:
        labels: Node labels for the subgraph nodes
    """
    if hasattr(data, 'y') and data.y is not None:
        return data.y[subset]
    return None


def prepare_node_classification_explanation(
    data: Data,
    target_node: int,
    model: torch.nn.Module,
    num_hops: int,
    device: torch.device
) -> Tuple[Data, torch.Tensor, int, torch.Tensor]:
    """
    Prepare all necessary data for explaining a single target node.
    
    Args:
        data: Full graph data
        target_node: Node to explain
        model: Trained GNN model
        num_hops: Number of hops for subgraph extraction
        device: Device to use
        
    Returns:
        subgraph: L-hop subgraph around target node
        subset: Original node IDs in subgraph
        target_new_id: Target node's ID in subgraph
        prediction: Model's prediction for the target node
    """
    # Extract L-hop subgraph
    subgraph, subset, target_new_id = extract_l_hop_subgraph(
        data, target_node, num_hops, relabel_nodes=True
    )
    
    # Move to device and get prediction
    data_device = data.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data_device.x, data_device.edge_index)
        
        # Handle different model output formats
        if logits.dim() == 1:
            # Single node output
            pred = logits
        elif logits.dim() == 2:
            # Node-level predictions
            pred = logits[target_node]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    return subgraph, subset, target_new_id, pred


def validate_subgraph_for_explanation(subgraph: Data) -> bool:
    """
    Validate that a subgraph is suitable for explanation.
    
    Args:
        subgraph: Subgraph data object
        
    Returns:
        valid: True if subgraph is valid
    """
    if subgraph.num_nodes == 0:
        return False
    if subgraph.edge_index.size(1) == 0:
        return False
    if not hasattr(subgraph, 'x') or subgraph.x is None:
        return False
    return True


def create_explanation_result(
    target_node: int,
    witness_subgraph: Data,
    original_subset: torch.Tensor,
    metrics: dict
) -> dict:
    """
    Package explanation results into a structured dictionary.
    
    Args:
        target_node: Original node ID that was explained
        witness_subgraph: The explanation subgraph (witness)
        original_subset: Mapping from witness nodes to original graph nodes
        metrics: Dictionary of computed metrics (fidelity, coverage, etc.)
        
    Returns:
        result: Structured explanation result
    """
    result = {
        'target_node': target_node,
        'witness': {
            'num_nodes': witness_subgraph.num_nodes,
            'num_edges': witness_subgraph.edge_index.size(1),
            'nodes': original_subset.tolist() if original_subset is not None else [],
            'edges': witness_subgraph.edge_index.t().tolist(),
        },
        'metrics': metrics,
    }
    
    return result


def batch_extract_l_hop_subgraphs(
    data: Data,
    target_nodes: List[int],
    num_hops: int
) -> List[Tuple[Data, torch.Tensor, int]]:
    """
    Extract L-hop subgraphs for multiple target nodes.
    
    Args:
        data: Full graph data
        target_nodes: List of target node IDs
        num_hops: Number of hops for subgraph extraction
        
    Returns:
        subgraphs: List of (subgraph, subset, target_new_id) tuples
    """
    subgraphs = []
    for target_node in target_nodes:
        try:
            subgraph, subset, target_new_id = extract_l_hop_subgraph(
                data, target_node, num_hops, relabel_nodes=True
            )
            if validate_subgraph_for_explanation(subgraph):
                subgraphs.append((subgraph, subset, target_new_id))
            else:
                print(f"Warning: Invalid subgraph for node {target_node}, skipping")
        except Exception as e:
            print(f"Error extracting subgraph for node {target_node}: {e}")
    
    return subgraphs


def print_subgraph_stats(subgraph: Data, target_node: int):
    """
    Print statistics about an extracted subgraph.
    
    Args:
        subgraph: Subgraph data object
        target_node: Original target node ID
    """
    print(f"\n{'='*60}")
    print(f"Subgraph Statistics for Target Node {target_node}")
    print(f"{'='*60}")
    print(f"  Nodes: {subgraph.num_nodes}")
    print(f"  Edges: {subgraph.edge_index.size(1)}")
    print(f"  Features: {subgraph.x.shape}")
    
    if hasattr(subgraph, 'y') and subgraph.y is not None:
        print(f"  Labels: {subgraph.y.shape}")
    
    if hasattr(subgraph, 'target_node_subgraph_id'):
        print(f"  Target node (subgraph ID): {subgraph.target_node_subgraph_id}")
    
    if hasattr(subgraph, 'original_node_ids'):
        print(f"  Original node IDs: {subgraph.original_node_ids.tolist()[:10]}...")
    
    print(f"{'='*60}\n")


__all__ = [
    'extract_l_hop_subgraph',
    'subgraph_to_full_node_mapping',
    'edges_to_full_graph_ids',
    'get_subgraph_node_labels',
    'prepare_node_classification_explanation',
    'validate_subgraph_for_explanation',
    'create_explanation_result',
    'batch_extract_l_hop_subgraphs',
    'print_subgraph_stats',
]
