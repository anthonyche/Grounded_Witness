#!/usr/bin/env python3
"""
Profile HeuChase on a single TreeCycle subgraph
找出到底是哪一步慢
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import time
import sys
import cProfile
import pstats
from io import StringIO

sys.path.append('src')
from heuchase import HeuChase
from constraints import get_constraints

# GCN model
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
    print("Profiling HeuChase on TreeCycle Subgraph")
    print("="*70)
    
    # Load data
    DATA_PATH = 'datasets/TreeCycle/treecycle_d5_bf15_n813616.pt'
    MODEL_PATH = 'models/TreeCycle_gcn_d5_bf15_n813616.pth'
    
    print("\n1. Loading data...")
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
    print(f"   ✓ Model loaded")
    
    # Extract a medium-sized subgraph (like Worker 14)
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
    )
    subgraph.task = 'node'
    subgraph.root = mapping.item()
    
    print(f"   ✓ Subgraph: {subgraph.num_nodes} nodes, {subgraph.edge_index.size(1)} edges")
    print(f"   ✓ Target node in subgraph: {subgraph.root}")
    
    # Load constraints
    print("\n4. Loading TreeCycle constraints...")
    CONSTRAINTS = get_constraints('TREECYCLE')
    print(f"   ✓ {len(CONSTRAINTS)} constraints")
    for c in CONSTRAINTS:
        print(f"      - {c['name']}")
    
    # Test with different m values
    print("\n5. Testing HeuChase with different m values...")
    print("-"*70)
    
    for m_value in [6, 10, 20]:
        print(f"\n--- Testing m={m_value} ---")
        
        # Create HeuChase explainer
        explainer = HeuChase(
            model=model,
            Sigma=CONSTRAINTS,
            L=2,
            k=6,
            B=8,
            m=m_value,
            noise_std=0.2,
            debug=False,  # No debug output for cleaner profiling
        )
        
        # Profile the _run() call
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.root))
        elapsed = time.time() - start_time
        
        profiler.disable()
        
        print(f"✓ Completed in {elapsed:.2f}s")
        print(f"  Found {len(S_k)} witnesses")
        print(f"  Coverage: {len(Sigma_star) if Sigma_star else 0} constraints")
        
        # Print top 10 time-consuming functions
        print(f"\n  Top 10 time-consuming functions:")
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        
        for line in s.getvalue().split('\n')[5:16]:  # Skip header
            if line.strip():
                print(f"    {line}")
        
        print("-"*70)
    
    print("\n" + "="*70)
    print("Profiling completed!")
    print("="*70)

if __name__ == "__main__":
    main()
