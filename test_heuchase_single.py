#!/usr/bin/env python3
"""
Test HeuChase on a single TreeCycle subgraph to measure runtime
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import time
import sys

sys.path.append('src')
from heuchase import HeuChase
from constraints import get_constraints

# Define GCN model (same as train_Treecycle.py)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def main():
    print("="*70)
    print("Testing HeuChase on a single TreeCycle subgraph")
    print("="*70)
    
    # Load data
    DATA_PATH = 'datasets/TreeCycle/treecycle_d5_bf15_n813616.pt'
    MODEL_PATH = 'models/TreeCycle_gcn_d5_bf15_n813616.pth'
    
    print("\n1. Loading TreeCycle graph...")
    data = torch.load(DATA_PATH)
    print(f"   ✓ {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    print("\n2. Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    in_channels = data.x.size(1)
    out_channels = len(torch.unique(data.y))
    hidden_channels = 32
    
    model = GCN(in_channels, hidden_channels, out_channels)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"   ✓ Model loaded (in={in_channels}, hidden={hidden_channels}, out={out_channels})")
    
    # Pick a test node (same as Worker 14 in your log)
    target_node = 660972
    num_hops = 2
    
    print(f"\n3. Extracting {num_hops}-hop subgraph for node {target_node}...")
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    
    subgraph = Data(
        x=data.x[subset],
        edge_index=edge_index,
        y=data.y[subset],
        subset=subset,
        target_node=mapping.item(),
    )
    subgraph.task = 'node'
    subgraph.root = mapping.item()
    
    print(f"   ✓ Subgraph: {subgraph.num_nodes} nodes, {subgraph.edge_index.size(1)} edges")
    print(f"   ✓ Target node in subgraph: {subgraph.target_node}")
    
    # Load constraints
    print("\n4. Loading TreeCycle constraints...")
    CONSTRAINTS = get_constraints('TREECYCLE')
    print(f"   ✓ {len(CONSTRAINTS)} constraints")
    
    # Create HeuChase explainer
    print("\n5. Creating HeuChase explainer...")
    explainer = HeuChase(
        model=model,
        Sigma=CONSTRAINTS,
        L=2,
        k=6,
        B=8,
        m=20,
        noise_std=0.2,
        debug=True,  # Enable debug output to see progress
    )
    print(f"   ✓ HeuChase initialized (B=8, m=20, k=6)")
    
    # Run HeuChase
    print("\n6. Running HeuChase._run()...")
    print("   (This may take a while for large subgraphs)")
    print(f"   Subgraph has {subgraph.edge_index.size(1)} edges to process")
    print("   Progress will be shown below:\n")
    print("-"*70)
    
    start_time = time.time()
    try:
        Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.target_node))
        elapsed = time.time() - start_time
        
        print("-"*70)
        print(f"\n✓ HeuChase completed in {elapsed:.2f}s")
        print(f"   Found {len(S_k)} witnesses")
        print(f"   Coverage: {len(Sigma_star) if Sigma_star else 0} constraints")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n✗ Interrupted after {elapsed:.2f}s")
        return 1
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\n✗ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
