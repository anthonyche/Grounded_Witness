"""
快速测试分布式解释器集成
验证 HeuChase, ApxChase, GNNExplainer 是否正确导入和运行
"""

import torch
import sys
sys.path.append('src')

from torch_geometric.data import Data
from Train_OGBN_HPC_MiniBatch import GCN_2_OGBN

print("=" * 70)
print("Testing Explainer Integration")
print("=" * 70)

# Step 1: Test imports
print("\n[1/5] Testing imports...")
try:
    from heuchase import HeuChase
    print("  ✓ HeuChase imported")
except Exception as e:
    print(f"  ✗ HeuChase import failed: {e}")

try:
    from apxchase import ApxChase
    print("  ✓ ApxChase imported")
except Exception as e:
    print(f"  ✗ ApxChase import failed: {e}")

try:
    from baselines import run_gnn_explainer_node
    print("  ✓ run_gnn_explainer_node imported")
except Exception as e:
    print(f"  ✗ run_gnn_explainer_node import failed: {e}")

# Step 2: Test model loading
print("\n[2/5] Testing model loading...")
try:
    MODEL_PATH = 'models/OGBN_Papers100M_epoch_20.pth'
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model = GCN_2_OGBN(
        input_dim=128,
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=172,
        dropout=0.5
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded: hidden_dim={checkpoint['hidden_dim']}")
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    exit(1)

# Step 3: Create dummy subgraph
print("\n[3/5] Creating dummy subgraph...")
try:
    # Small dummy graph: 10 nodes, 20 edges
    num_nodes = 10
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        [1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 0, 7, 1]
    ], dtype=torch.long)
    
    x = torch.randn(num_nodes, 128)  # 128-dim features
    y = torch.randint(0, 172, (num_nodes,))  # 172 classes
    
    subgraph = Data(x=x, edge_index=edge_index, y=y)
    subgraph.target_node = 0  # Explain node 0
    
    print(f"  ✓ Dummy subgraph created: {num_nodes} nodes, {edge_index.size(1)} edges")
except Exception as e:
    print(f"  ✗ Subgraph creation failed: {e}")
    exit(1)

# Step 4: Test HeuChase
print("\n[4/5] Testing HeuChase...")
try:
    explainer_config = {
        'Sigma': None,
        'L': 2,
        'k': 5,
        'B': 3,
        'm': 3,  # Fewer candidates for quick test
        'noise_std': 1e-3,
    }
    
    explainer = HeuChase(
        model=model,
        Sigma=explainer_config['Sigma'],
        L=explainer_config['L'],
        k=explainer_config['k'],
        B=explainer_config['B'],
        m=explainer_config['m'],
        noise_std=explainer_config['noise_std'],
        debug=False,
    )
    
    Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.target_node))
    print(f"  ✓ HeuChase ran successfully: {len(S_k)} witnesses, coverage={len(Sigma_star)}")
except Exception as e:
    print(f"  ✗ HeuChase test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Test ApxChase
print("\n[5/5] Testing ApxChase...")
try:
    explainer_config = {
        'Sigma': None,
        'L': 2,
        'k': 5,
        'B': 3,
    }
    
    explainer = ApxChase(
        model=model,
        Sigma=explainer_config['Sigma'],
        L=explainer_config['L'],
        k=explainer_config['k'],
        B=explainer_config['B'],
        debug=False,
    )
    
    Sigma_star, S_k = explainer._run(H=subgraph, root=int(subgraph.target_node))
    print(f"  ✓ ApxChase ran successfully: {len(S_k)} witnesses, coverage={len(Sigma_star)}")
except Exception as e:
    print(f"  ✗ ApxChase test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Test GNNExplainer (optional, slower)
print("\n[6/6] Testing GNNExplainer (optional)...")
try:
    result = run_gnn_explainer_node(
        model=model,
        data=subgraph,
        target_node=int(subgraph.target_node),
        epochs=10,  # Very few epochs for quick test
        lr=0.01,
        device=torch.device('cpu'),
    )
    print(f"  ✓ GNNExplainer ran successfully: pred={result['pred']}, success={result['success']}")
except Exception as e:
    print(f"  ⚠ GNNExplainer test failed (expected, may need more epochs): {e}")

print("\n" + "=" * 70)
print("Integration Test Completed!")
print("=" * 70)
print("\nNext steps:")
print("1. Submit distributed benchmark: sbatch run_ogbn_distributed_bench.slurm")
print("2. Monitor progress: watch -n 60 'squeue -u $USER'")
print("3. Check logs: tail -f results/ogbn_distributed/*.log")
