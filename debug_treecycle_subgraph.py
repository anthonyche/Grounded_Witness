"""
Debug TreeCycle subgraph properties to understand Edmonds failure
"""
import torch
import networkx as nx
from torch_geometric.data import Data

# Load TreeCycle data
print("Loading TreeCycle dataset...")
data = torch.load('datasets/TreeCycle/treecycle_d5_bf15_n813616.pt')
print(f"Full graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

# Load subgraph cache to get actual subgraph
import pickle
cache_path = 'cache/treecycle/subgraph_485671_L2.pkl'
print(f"\nLoading cached subgraph: {cache_path}")
with open(cache_path, 'rb') as f:
    subgraph = pickle.load(f)

print(f"\nSubgraph properties:")
print(f"  Nodes: {subgraph.num_nodes}")
print(f"  Edges: {subgraph.edge_index.size(1)}")
print(f"  Target node: {subgraph.target_node}")

# Check if it's a DAG or has cycles
edge_index = subgraph.edge_index
num_nodes = subgraph.num_nodes
num_edges = edge_index.size(1)

# Build undirected graph
G_undirected = nx.Graph()
for i in range(num_edges):
    u = int(edge_index[0, i].item())
    v = int(edge_index[1, i].item())
    G_undirected.add_edge(u, v)

print(f"\nUndirected graph analysis:")
print(f"  Connected: {nx.is_connected(G_undirected)}")
print(f"  Components: {nx.number_connected_components(G_undirected)}")
print(f"  Density: {nx.density(G_undirected):.4f}")

# Check for tree structure
if nx.is_connected(G_undirected):
    is_tree = nx.is_tree(G_undirected)
    print(f"  Is tree: {is_tree}")
    if not is_tree:
        num_cycles = len(nx.cycle_basis(G_undirected))
        print(f"  Cycles found: {num_cycles}")

# Build DIRECTED graph (like _candidate_by_edmonds does)
G_directed = nx.DiGraph()
for n in range(num_nodes):
    G_directed.add_node(n)

for i in range(num_edges):
    u = int(edge_index[0, i].item())
    v = int(edge_index[1, i].item())
    # Add BOTH directions (like heuchase does)
    G_directed.add_edge(u, v, weight=1.0)
    G_directed.add_edge(v, u, weight=1.0)

print(f"\nDirected graph analysis:")
print(f"  Nodes: {G_directed.number_of_nodes()}")
print(f"  Edges: {G_directed.number_of_edges()}")
print(f"  Weakly connected: {nx.is_weakly_connected(G_directed)}")
print(f"  Strongly connected: {nx.is_strongly_connected(G_directed)}")

# Check if it's a DAG
is_dag = nx.is_directed_acyclic_graph(G_directed)
print(f"  Is DAG: {is_dag}")

if not is_dag:
    # Find strongly connected components
    sccs = list(nx.strongly_connected_components(G_directed))
    print(f"  Strongly connected components: {len(sccs)}")
    print(f"    Largest SCC size: {max(len(scc) for scc in sccs)}")

# Try to run Edmonds
print(f"\nTrying nx.maximum_spanning_arborescence...")
import time
start = time.time()
try:
    Ar = nx.maximum_spanning_arborescence(G_directed, attr='weight', default=0)
    elapsed = time.time() - start
    print(f"✓ Success in {elapsed:.2f}s")
    print(f"  Result: {Ar.number_of_nodes()} nodes, {Ar.number_of_edges()} edges")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ FAILED in {elapsed:.2f}s")
    print(f"  Error: {e}")
    print(f"\n  This is why HeuChase hangs!")
