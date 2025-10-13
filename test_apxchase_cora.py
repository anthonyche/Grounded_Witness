"""
Test ApxChase on Cora citation network
"""

import os
import sys
import torch
import yaml
import time
sys.path.insert(0, 'src')

from src.model import get_model
from src.utils import dataset_func
from src.apxchase import ApxChase
from src.constraints import get_constraints

print("=" * 70)
print("ApxChase on Cora Citation Network")
print("=" * 70)
print()

# Load config
print("Step 1: Load configuration")
print("-" * 70)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"  Data: {config['data_name']}")
print(f"  Model: {config['model_name']}")
print(f"  L (hops): {config['L']}")
print(f"  k (window): {config['k']}")
print(f"  Budget: {config['Budget']}")
print()

# Load model
print("Step 2: Load trained model")
print("-" * 70)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = get_model(config).to(device)
model_path = f"models/{config['data_name']}_{config['model_name']}_model.pth"
print(f"  Loading: {model_path}")

checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    model.load_state_dict(checkpoint)
    print(f"  ✅ Loaded checkpoint (direct state_dict)")

model.eval()
print(f"  Model class: {model.__class__.__name__}")
print()

# Load dataset
print("Step 3: Load Cora dataset")
print("-" * 70)
dataset_resource = dataset_func(config)
if isinstance(dataset_resource, dict):
    data = dataset_resource['data']
    print(f"  ✅ Loaded from dict")
else:
    data = dataset_resource
    print(f"  ✅ Loaded directly")

print(f"  Nodes: {data.num_nodes:,}")
print(f"  Edges: {data.edge_index.size(1):,}")
print(f"  Features: {data.x.size(1)}")
print(f"  Classes: {data.y.max().item() + 1}")
print(f"  Train nodes: {data.train_mask.sum().item():,}")
print(f"  Val nodes: {data.val_mask.sum().item():,}")
print(f"  Test nodes: {data.test_mask.sum().item():,}")
print()

# Class names
class_names = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory"
]

# Select target node
print("Step 4: Select target node")
print("-" * 70)
test_indices = torch.where(data.test_mask)[0]
target_node = int(test_indices[0].item())
print(f"  First test node ID: {target_node}")
print(f"  True class: {data.y[target_node].item()} ({class_names[data.y[target_node].item()]})")

# Get ground truth prediction
with torch.no_grad():
    data_cpu = data.clone()
    out = model(data_cpu.x, data_cpu.edge_index)
    pred = out.argmax(dim=-1)
    target_pred = pred[target_node].item()
    
print(f"  Predicted class: {target_pred} ({class_names[target_pred]})")
print(f"  Correct: {'✅ Yes' if target_pred == data.y[target_node].item() else '❌ No'}")
print()

# Load constraints
print("Step 5: Load TGD constraints")
print("-" * 70)
constraints = get_constraints(config['data_name'])
print(f"  ✅ Loaded {len(constraints)} constraints for Cora:")
for i, c in enumerate(constraints):
    name = c.get('name', f'constraint_{i}')
    head_edges = len(c['head']['edges'])
    body_edges = len(c['body']['edges'])
    print(f"    [{i+1}] {name}: {head_edges}-edge HEAD → {body_edges}-edge BODY")
print()

# Initialize ApxChase
print("Step 6: Initialize ApxChase")
print("-" * 70)
print(f"  Creating ApxChase with:")
print(f"    L={config['L']}, k={config['k']}, B={config['Budget']}")
print(f"    alpha={config.get('alpha', 1.0)}, beta={config.get('beta', 1.0)}, gamma={config.get('gamma', 1.0)}")

apxchase = ApxChase(
    model=model,
    Sigma=constraints,
    L=config['L'],
    k=config['k'],
    B=config['Budget'],
    alpha=config.get('alpha', 1.0),
    beta=config.get('beta', 1.0),
    gamma=config.get('gamma', 1.0),
    debug=True
)
print(f"  ✅ ApxChase initialized")
print()

# Run ApxChase
print("Step 7: Run ApxChase.explain_node()")
print("-" * 70)
print(f"  Target node: {target_node} (class: {class_names[data.y[target_node].item()]})")
print(f"  Starting explanation...")
print()

overall_start_time = time.time()
start_time = time.time()

try:
    # Attach y_ref for verification
    data.y_ref = pred  # Full graph predictions
    
    sigma_star, S_k = apxchase.explain_node(data, target_node)
    
    elapsed = time.time() - start_time
    overall_elapsed = time.time() - overall_start_time
    
    print()
    print("=" * 70)
    print(f"✅ ApxChase completed in {elapsed:.2f}s")
    print(f"   Overall runtime: {overall_elapsed:.2f}s")
    print("=" * 70)
    print()
    print(f"Results:")
    print(f"  Grounded constraints (Σ*): {len(sigma_star)}")
    print(f"  Explanation subgraphs (S_k): {len(S_k)}")
    print()
    
    if sigma_star:
        print(f"  Grounded constraints:")
        for name in sorted(sigma_star):
            print(f"    - {name}")
    else:
        print(f"  ⚠️  No constraints grounded (empty Σ*)")
    
    print()
    if S_k:
        print(f"  Explanation subgraphs:")
        for i, subgraph in enumerate(S_k[:5]):  # Show first 5
            num_nodes = subgraph.num_nodes
            num_edges = subgraph.edge_index.size(1)
            score = getattr(subgraph, '_score', 'N/A')
            print(f"    [{i}] |V|={num_nodes}, |E|={num_edges}, score={score}")
        if len(S_k) > 5:
            print(f"    ... and {len(S_k) - 5} more")
    else:
        print(f"  ⚠️  No explanation subgraphs found")
    
    print()
    print("=" * 70)
    print("✅ TEST PASSED: ApxChase runs successfully on Cora!")
    print(f"   Total runtime: {overall_elapsed:.2f}s (ApxChase: {elapsed:.2f}s)")
    print("=" * 70)
        
except Exception as e:
    elapsed = time.time() - start_time
    overall_elapsed = time.time() - overall_start_time
    print()
    print("=" * 70)
    print(f"❌ TEST FAILED after {elapsed:.2f}s (overall: {overall_elapsed:.2f}s)")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Check the error trace above for details.")

print()
print("Test completed.")
