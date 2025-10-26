"""
Quick test: ApxChase on a synthetic TreeCycle-like subgraph
Verify it doesn't hang like HeuChase does
"""
import torch
import time
from torch_geometric.data import Data

# Simulate a TreeCycle subgraph
print("Creating synthetic TreeCycle-like subgraph...")
num_nodes = 100
num_edges = 200

torch.manual_seed(42)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
x = torch.randn(num_nodes, 32)  # 32-dim features
y = torch.randint(0, 4, (num_nodes,))  # 4 classes

subgraph = Data(x=x, edge_index=edge_index, y=y)
subgraph.target_node = 0
subgraph.num_nodes = num_nodes

print(f"Subgraph: {num_nodes} nodes, {num_edges} edges")

# Create dummy model
from torch.nn import Module
class DummyGCN(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, edge_index):
        return torch.randn(x.size(0), 4)  # 4 classes

model = DummyGCN()
model.eval()

# Create dummy constraints
CONSTRAINTS = [
    {'name': 'dummy1', 'type': 'tgd', 'body': [], 'head': []},
    {'name': 'dummy2', 'type': 'tgd', 'body': [], 'head': []},
]

print("\nTesting ApxChase...")
try:
    from src.apxchase import ApxChase
    
    apx = ApxChase(
        model=model,
        Sigma=CONSTRAINTS,
        L=2,
        k=10,
        B=5,
        debug=True
    )
    
    print("Running apxchase._run()...")
    start = time.time()
    Sigma_star, S_k = apx._run(H=subgraph, root=0)
    elapsed = time.time() - start
    
    print(f"\n✓ ApxChase completed in {elapsed:.2f}s")
    print(f"  Coverage: {len(Sigma_star)}")
    print(f"  Witnesses: {len(S_k)}")
    
except Exception as e:
    print(f"\n✗ ApxChase failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting HeuChase (should hang/fail)...")
try:
    from src.heuchase import HeuChase
    
    heu = HeuChase(
        model=model,
        Sigma=CONSTRAINTS,
        L=2,
        k=10,
        B=5,
        m=6,
        noise_std=1e-3,
        debug=True
    )
    
    print("Running heuchase._run()...")
    start = time.time()
    Sigma_star, S_k = heu._run(H=subgraph, root=0)
    elapsed = time.time() - start
    
    print(f"\n✓ HeuChase completed in {elapsed:.2f}s")
    print(f"  Coverage: {len(Sigma_star)}")
    print(f"  Witnesses: {len(S_k)}")
    
except Exception as e:
    elapsed = time.time() - start
    print(f"\n✗ HeuChase failed after {elapsed:.2f}s: {e}")
