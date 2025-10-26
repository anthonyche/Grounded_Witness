"""
Test Edmonds algorithm speed on TreeCycle subgraphs
"""
import torch
import time
import networkx as nx
from torch_geometric.data import Data

def test_edmonds_on_treecycle():
    """Test how long Edmonds takes on a TreeCycle-like subgraph"""
    
    # Simulate a 638-node, 1258-edge subgraph (like Worker 12's task)
    num_nodes = 638
    num_edges = 1258
    
    print(f"Testing Edmonds on {num_nodes} nodes, {num_edges} edges...")
    print(f"Density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.4f}")
    
    # Create random edge_index
    torch.manual_seed(42)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Build directed multigraph
    G = nx.DiGraph()
    for n in range(num_nodes):
        G.add_node(n)
    
    # Add edges with random weights
    for idx in range(num_edges):
        u = int(edge_index[0, idx].item())
        v = int(edge_index[1, idx].item())
        w_uv = torch.rand(1).item()
        w_vu = torch.rand(1).item()
        G.add_edge(u, v, weight=w_uv, _eid=idx)
        G.add_edge(v, u, weight=w_vu, _eid=idx)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Time Edmonds algorithm
    print("\nRunning nx.maximum_spanning_arborescence...")
    start = time.time()
    try:
        Ar = nx.maximum_spanning_arborescence(G, attr='weight', default=0)
        elapsed = time.time() - start
        print(f"✓ Edmonds completed in {elapsed:.2f}s")
        print(f"  Result: {Ar.number_of_nodes()} nodes, {Ar.number_of_edges()} edges")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Edmonds failed after {elapsed:.2f}s: {e}")
    
    return elapsed

def test_multiple_runs():
    """Test m=6 runs like HeuChase does"""
    print("\n" + "="*60)
    print("Testing m=6 Edmonds runs (like HeuChase)")
    print("="*60)
    
    total_time = 0
    for i in range(6):
        print(f"\nRun {i+1}/6:")
        t = test_edmonds_on_treecycle()
        total_time += t
    
    print("\n" + "="*60)
    print(f"Total time for m=6 runs: {total_time:.2f}s")
    print(f"Average per run: {total_time/6:.2f}s")
    print("="*60)

if __name__ == '__main__':
    # Quick test
    print("Quick test (single run):")
    test_edmonds_on_treecycle()
    
    # Full test
    test_multiple_runs()
