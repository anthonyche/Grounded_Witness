"""
测试 HeuChase 在 TreeCycle 子图上的速度
"""
import torch
import time
import sys
sys.path.append('src')

from heuchase import HeuChase
from constraints import get_constraints
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.last_node_embeddings = None
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        self.last_node_embeddings = x
        x = self.conv2(x, edge_index)
        return x
    
    def get_node_embeddings(self, data):
        with torch.no_grad():
            x = self.conv1(data.x, data.edge_index)
            x = F.relu(x)
            return x

print("="*70)
print("Testing HeuChase Speed on TreeCycle Subgraphs")
print("="*70)

# Create a synthetic small graph for testing
print("\n1. Creating synthetic test graph...")
num_nodes = 1000
num_edges = 1500
edge_index = torch.randint(0, num_nodes, (2, num_edges))
x = torch.randn(num_nodes, 16)
y = torch.randint(0, 3, (num_nodes,))

data = Data(x=x, edge_index=edge_index, y=y)
print(f"   Graph: {num_nodes} nodes, {num_edges} edges")

# Extract a 2-hop subgraph
print("\n2. Extracting 2-hop subgraph from node 0...")
subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
    node_idx=0,
    num_hops=2,
    edge_index=data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
)

subgraph = Data(
    x=data.x[subset].clone(),
    edge_index=edge_index_sub.clone(),
    y=data.y[subset].clone(),
    subset=subset.clone(),
    target_node=mapping.item(),
)
subgraph.task = 'node'
subgraph.root = mapping.item()
subgraph._target_node_subgraph_id = mapping.item()
subgraph._nodes_in_full = subset.clone()
subgraph.num_nodes = subset.size(0)
subgraph.E_base = edge_index_sub.size(1)

print(f"   Subgraph: {subgraph.num_nodes} nodes, {subgraph.edge_index.size(1)} edges")

# Create model
print("\n3. Creating GCN model...")
model = GCN(in_channels=16, hidden_channels=32, out_channels=3)
model.eval()
print("   Model created ✓")

# Load constraints
print("\n4. Loading TreeCycle constraints...")
constraints = get_constraints('TREECYCLE')
print(f"   Loaded {len(constraints)} constraints")

# Initialize HeuChase
print("\n5. Initializing HeuChase...")
start_init = time.time()
explainer = HeuChase(
    model=model,
    Sigma=constraints,
    L=2,
    k=10,
    B=50,
    m=6,
    noise_std=1e-3,
    debug=True  # Enable debug output
)
init_time = time.time() - start_init
print(f"   HeuChase initialized in {init_time:.3f}s ✓")

# Run HeuChase
print("\n6. Running HeuChase._run() on subgraph...")
print(f"   Subgraph size: {subgraph.num_nodes} nodes, {subgraph.edge_index.size(1)} edges")
print(f"   Target node: {subgraph.root}")
print("-" * 70)

start_run = time.time()
try:
    Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.root))
    run_time = time.time() - start_run
    
    print("-" * 70)
    print(f"\n7. Results:")
    print(f"   Runtime: {run_time:.3f}s")
    print(f"   Witnesses found: {len(S_k)}")
    print(f"   Coverage: {len(Sigma_star)}")
    print(f"   Success: {'✓' if len(S_k) > 0 else '✗'}")
    
    if run_time > 5:
        print(f"\n   ⚠️  WARNING: Runtime > 5s, this is slow!")
        print(f"      For 1400-edge subgraphs, this could be 5-10x slower")
    else:
        print(f"\n   ✓ Performance looks good!")
        
except Exception as e:
    run_time = time.time() - start_run
    print("-" * 70)
    print(f"\n7. ERROR after {run_time:.3f}s:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test completed")
print("="*70)
