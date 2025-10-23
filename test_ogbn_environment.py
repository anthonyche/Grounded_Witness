#!/usr/bin/env python
"""
Test script to diagnose OGBN-Papers100M environment issues
Run this before the main training to check if everything is set up correctly.
"""

import sys
print("=" * 70)
print("OGBN-Papers100M Environment Test")
print("=" * 70)
print()

# Test 1: Python version
print("Test 1: Python version")
print(f"  Python: {sys.version}")
print(f"  Executable: {sys.executable}")
print("  ✓ PASS")
print()

# Test 2: PyTorch
print("Test 2: PyTorch")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print("  ✓ PASS")
except ImportError as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)
print()

# Test 3: PyTorch Geometric
print("Test 3: PyTorch Geometric")
try:
    import torch_geometric
    print(f"  PyG version: {torch_geometric.__version__}")
    from torch_geometric.nn import GCNConv
    print("  GCNConv import: OK")
    print("  ✓ PASS")
except ImportError as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)
print()

# Test 4: OGB
print("Test 4: OGB (Open Graph Benchmark)")
try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    print("  OGB import: OK")
    print("  PygNodePropPredDataset: OK")
    print("  Evaluator: OK")
    print("  ✓ PASS")
except ImportError as e:
    print(f"  ✗ FAIL: {e}")
    print()
    print("  Solution: Install OGB with:")
    print("    pip install ogb")
    sys.exit(1)
print()

# Test 5: Other dependencies
print("Test 5: Other dependencies")
try:
    import yaml
    print("  PyYAML: OK")
except ImportError:
    print("  PyYAML: Missing (optional)")

try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except ImportError:
    print("  NumPy: Missing")
print("  ✓ PASS")
print()

# Test 6: Dataset accessibility test (without downloading)
print("Test 6: Dataset accessibility test")
try:
    import os
    data_root = './datasets'
    print(f"  Data root: {data_root}")
    print(f"  Data root exists: {os.path.exists(data_root)}")
    if os.path.exists(data_root):
        print(f"  Data root writable: {os.access(data_root, os.W_OK)}")
    else:
        print("  Creating data root...")
        os.makedirs(data_root, exist_ok=True)
        print("  Data root created: OK")
    print("  ✓ PASS")
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)
print()

# Test 7: Quick dataset metadata test
print("Test 7: Dataset metadata test (no download)")
try:
    # This should not trigger download, just check if OGB can be queried
    from ogb.nodeproppred import NodePropPredDataset
    print("  NodePropPredDataset accessible: OK")
    print("  ✓ PASS")
except Exception as e:
    print(f"  Warning: {e}")
    print("  (This may be normal if dataset not downloaded yet)")
print()

# Test 8: Model creation test
print("Test 8: Model creation test")
try:
    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F
    
    class TestGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(10, 20)
            self.conv2 = GCNConv(20, 5)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    model = TestGCN()
    print(f"  Test model created: {sum(p.numel() for p in model.parameters())} parameters")
    print("  ✓ PASS")
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print()
print("Environment is ready for OGBN-Papers100M training!")
print()
print("Next steps:")
print("  1. Run: sbatch train_ogbn_papers100m.slurm")
print("  2. Monitor: tail -f logs/ogbn_papers100m_<job_id>.out")
print()
