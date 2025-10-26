"""
快速测试单个 HeuChase 任务，判断是否会卡住
"""

import torch
import time
import sys
sys.path.append('src')

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from heuchase import HeuChase
from constraints import get_constraints

# Import GCN model
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

print("Loading TreeCycle graph...")
data = torch.load('datasets/TreeCycle/treecycle_d5_bf15_n813616.pt')
print(f"Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

print("\nLoading model...")
checkpoint = torch.load('models/TreeCycle_gcn_d5_bf15_n813616.pth', map_location='cpu')
in_channels = data.x.size(1)
out_channels = len(torch.unique(data.y))
hidden_channels = 32

model = GCN(in_channels, hidden_channels, out_channels)
model.load_state_dict(checkpoint)
model.eval()
print(f"Model loaded (in={in_channels}, hidden={hidden_channels}, out={out_channels})")

print("\nLoading constraints...")
CONSTRAINTS = get_constraints('TREECYCLE')
print(f"Loaded {len(CONSTRAINTS)} constraints")

# Test with a specific node that was hanging
test_node = 485671  # From Worker 12
print(f"\nExtracting subgraph for node {test_node}...")

subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=test_node,
    num_hops=2,
    edge_index=data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
)

subgraph = Data(
    x=data.x[subset].clone().detach(),
    edge_index=edge_index.clone().detach(),
    y=data.y[subset].clone().detach(),
    subset=subset.clone().detach(),
    target_node=mapping.item(),
)
subgraph.task = 'node'
subgraph.root = mapping.item()
subgraph._target_node_subgraph_id = mapping.item()

print(f"Subgraph: {subgraph.num_nodes} nodes, {subgraph.edge_index.size(1)} edges")
print(f"Target node in subgraph: {subgraph.target_node}")

print("\nInitializing HeuChase...")
explainer = HeuChase(
    model=model,
    Sigma=CONSTRAINTS,
    L=2,
    k=10,
    B=8,
    m=6,
    noise_std=1e-3,
    debug=True,  # Enable debug mode!
)
print("HeuChase initialized")

print(f"\n{'='*60}")
print("Calling HeuChase._run()...")
print(f"{'='*60}")
sys.stdout.flush()

start_time = time.time()

# Add timeout monitoring
import threading
def print_progress():
    elapsed = 0
    while True:
        time.sleep(5)
        elapsed += 5
        print(f"[{elapsed}s] Still running...", flush=True)

progress_thread = threading.Thread(target=print_progress, daemon=True)
progress_thread.start()

try:
    Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.target_node))
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"COMPLETED in {elapsed:.2f}s")
    print(f"{'='*60}")
    print(f"Found {len(S_k)} witnesses")
    print(f"Coverage: {len(Sigma_star) if Sigma_star else 0} constraints")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ERROR after {elapsed:.2f}s: {e}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
