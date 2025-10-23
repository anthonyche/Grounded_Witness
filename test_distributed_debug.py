"""
Debug script for distributed benchmark
Test with minimal settings to identify issues
"""

import sys
sys.path.append('src')

from benchmark_ogbn_distributed import *

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("="*70)
    print("DEBUG: Minimal Test for Distributed Benchmark")
    print("="*70)
    
    # Test configuration
    NUM_TEST_NODES = 5  # Very small for debugging
    NUM_WORKERS = 2
    NUM_HOPS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTest Configuration:")
    print(f"  Nodes: {NUM_TEST_NODES}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Hops: {NUM_HOPS}")
    print(f"  Device: {DEVICE}")
    print()
    
    # Load constraints
    print("Loading constraints...")
    from constraints import get_constraints
    CONSTRAINTS = get_constraints('OGBN-PAPERS100M')
    print(f"  Loaded {len(CONSTRAINTS)} constraints")
    
    # Explainer configs
    EXPLAINER_CONFIGS = {
        'gnnexplainer': {
            'epochs': 100,
            'lr': 0.01,
        }
    }
    
    # Load dataset
    print("\nLoading OGBN-Papers100M dataset...")
    try:
        dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='./datasets')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        
        print(f"  Dataset loaded!")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.edge_index.size(1):,}")
        
        # Memory check
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        print(f"  Memory usage: {mem_gb:.2f} GB")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Sample test nodes
    test_nodes = split_idx['test'].numpy()
    np.random.seed(42)
    sampled_nodes = np.random.choice(test_nodes, size=NUM_TEST_NODES, replace=False)
    
    # Convert to Python int to avoid numpy.int64 issues
    sampled_nodes = [int(node) for node in sampled_nodes]
    
    print(f"\nSampled nodes: {sampled_nodes}")
    
    # Test with GNNExplainer only
    print("\n" + "="*70)
    print("Testing GNNExplainer with minimal settings")
    print("="*70)
    
    try:
        result = run_distributed_benchmark(
            data=data,
            model_path='models/OGBN_Papers100M_epoch_20.pth',
            node_ids=sampled_nodes,
            explainer_name='gnnexplainer',
            num_workers=NUM_WORKERS,
            explainer_config=EXPLAINER_CONFIGS['gnnexplainer'],
            num_hops=NUM_HOPS,
            device=DEVICE
        )
        
        print("\n" + "="*70)
        print("SUCCESS! Test completed")
        print("="*70)
        print(f"Total time: {result['total_time']:.2f}s")
        print(f"Parallel time: {result['parallel_time']:.2f}s")
        print(f"Speedup: {result['worker_time_mean'] / result['parallel_time']:.2f}x")
        
    except Exception as e:
        print("\n" + "="*70)
        print("FAILED! Error during test")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
