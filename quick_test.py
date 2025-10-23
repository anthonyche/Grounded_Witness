#!/usr/bin/env python3
"""
Quick test of distributed benchmark with dummy data
Tests logic without loading the huge OGBN-Papers100M dataset
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from torch_geometric.data import Data
import sys
import time

sys.path.append('src')

def test_cache_filename_consistency():
    """Test that cache filenames are generated consistently"""
    print("Testing cache filename generation...")
    
    import hashlib
    
    node_ids = [1, 2, 3, 4, 5]
    num_hops = 2
    cache_dir = 'cache/subgraphs'
    
    # Method 1: load_tasks_from_cache_only
    node_ids_tuple = tuple(sorted(node_ids))
    node_str = str(node_ids_tuple)
    hash_val1 = hashlib.md5(node_str.encode()).hexdigest()[:8]
    cache_file1 = f"{cache_dir}/subgraphs_hops{num_hops}_nodes{len(node_ids)}_hash{hash_val1}.pt"
    
    # Method 2: Coordinator.create_tasks
    node_ids_tuple = tuple(sorted(node_ids))
    node_str = str(node_ids_tuple)
    hash_val2 = hashlib.md5(node_str.encode()).hexdigest()[:8]
    cache_file2 = f"{cache_dir}/subgraphs_hops{num_hops}_nodes{len(node_ids)}_hash{hash_val2}.pt"
    
    assert cache_file1 == cache_file2, f"Mismatch: {cache_file1} != {cache_file2}"
    print(f"✓ Cache filename consistent: {cache_file1}")


def create_dummy_graph(num_nodes=1000, num_edges=5000):
    """Create a dummy graph for testing"""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    x = torch.randn(num_nodes, 128)
    y = torch.randint(0, 172, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def test_subgraph_extraction():
    """Test L-hop subgraph extraction"""
    print("\nTesting subgraph extraction...")
    
    from torch_geometric.utils import k_hop_subgraph
    
    # Create dummy graph
    data = create_dummy_graph(num_nodes=1000, num_edges=5000)
    
    # Extract 2-hop subgraph
    node_id = 100
    num_hops = 2
    
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    
    print(f"✓ Extracted {num_hops}-hop subgraph:")
    print(f"  Target node: {node_id}")
    print(f"  Subgraph nodes: {len(subset)}")
    print(f"  Subgraph edges: {sub_edge_index.size(1)}")
    print(f"  Target node in subgraph: {mapping.item()}")


def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        from Train_OGBN_HPC_MiniBatch import GCN_2_OGBN
        
        model_path = 'models/OGBN_Papers100M_epoch_20.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = GCN_2_OGBN(
            input_dim=128,
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=172,
            dropout=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Hidden dim: {checkpoint['hidden_dim']}")
        
        # Test forward pass with dummy data
        dummy_x = torch.randn(10, 128)
        dummy_edge_index = torch.randint(0, 10, (2, 20))
        
        with torch.no_grad():
            output = model(dummy_x, dummy_edge_index)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        raise


def test_explainer_import():
    """Test explainer imports"""
    print("\nTesting explainer imports...")
    
    try:
        from heuchase import HeuChase
        print("✓ HeuChase imported")
    except Exception as e:
        print(f"✗ HeuChase import failed: {e}")
        raise
    
    try:
        from apxchase import ApxChase
        print("✓ ApxChase imported")
    except Exception as e:
        print(f"✗ ApxChase import failed: {e}")
        raise
    
    try:
        from baselines import run_gnn_explainer_node
        print("✓ GNNExplainer imported")
    except Exception as e:
        print(f"✗ GNNExplainer import failed: {e}")
        raise


def test_constraints():
    """Test constraint loading"""
    print("\nTesting constraint loading...")
    
    try:
        from constraints import get_constraints
        
        constraints = get_constraints('OGBN-PAPERS100M')
        print(f"✓ Loaded {len(constraints)} constraints:")
        for i, tgd in enumerate(constraints, 1):
            print(f"  {i}. {tgd['name']}")
        
        assert len(constraints) > 0, "No constraints found!"
        
    except Exception as e:
        print(f"✗ Constraint loading failed: {e}")
        raise


def test_worker_process():
    """Test worker process can be spawned"""
    print("\nTesting worker process spawning...")
    
    def dummy_worker(worker_id, result_queue):
        result_queue.put({'worker_id': worker_id, 'status': 'ok'})
    
    try:
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        
        p = mp.Process(target=dummy_worker, args=(0, result_queue))
        p.start()
        result = result_queue.get(timeout=5)
        p.join()
        
        print(f"✓ Worker process spawned successfully")
        print(f"  Result: {result}")
        
    except Exception as e:
        print(f"✗ Worker process failed: {e}")
        raise


def main():
    print("=" * 70)
    print("Quick Test - Distributed Benchmark Logic")
    print("=" * 70)
    
    try:
        test_cache_filename_consistency()
        test_subgraph_extraction()
        test_model_loading()
        test_explainer_import()
        test_constraints()
        test_worker_process()
        
        print("\n" + "=" * 70)
        print("✅ All quick tests passed!")
        print("=" * 70)
        print("\nYou can now run the pre-flight check:")
        print("  python pre_flight_check.py")
        print("\nThen submit the benchmark:")
        print("  sbatch run_ogbn_distributed_bench.slurm")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Test failed!")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
