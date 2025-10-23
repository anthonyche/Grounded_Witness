"""
快速测试分布式基准 - 只测试 5 个节点，2 个 workers
验证完整的 Coordinator-Worker 架构
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import sys
import os

sys.path.append('src')

from ogb.nodeproppred import PygNodePropPredDataset
from benchmark_ogbn_distributed import run_distributed_benchmark
from constraints import get_constraints

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    print("=" * 70)
    print("Quick Distributed Benchmark Test (5 nodes, 2 workers)")
    print("=" * 70)
    
    # Configuration
    NUM_SAMPLE_NODES = 5  # Very small for quick test
    NUM_HOPS = 2
    NUM_WORKERS = 2
    MODEL_PATH = 'models/OGBN_Papers100M_epoch_20.pth'
    DEVICE = 'cpu'
    
    # Load constraints
    print("\n[0/3] Loading constraints for OGBN-Papers100M...")
    CONSTRAINTS = get_constraints('OGBN-PAPERS100M')
    print(f"  ✓ Loaded {len(CONSTRAINTS)} constraints:")
    for tgd in CONSTRAINTS:
        print(f"    - {tgd['name']}")
    
    # Explainer configurations
    EXPLAINER_CONFIGS = {
        'heuchase': {
            'Sigma': CONSTRAINTS,  # Use real constraints
            'L': NUM_HOPS,
            'k': 5,
            'B': 3,
            'm': 3,  # Fewer candidates for quick test
            'noise_std': 1e-3,
        },
        'apxchase': {
            'Sigma': CONSTRAINTS,  # Use real constraints
            'L': NUM_HOPS,
            'k': 5,
            'B': 3,
        },
        'gnnexplainer': {
            'epochs': 50,  # Fewer epochs for quick test
            'lr': 0.01,
        }
    }
    
    # Load dataset
    print("\n[1/3] Loading OGBN-Papers100M dataset...")
    try:
        dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='./datasets')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        print(f"  ✓ Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        exit(1)
    
    # Sample nodes
    print(f"\n[2/3] Sampling {NUM_SAMPLE_NODES} test nodes...")
    test_nodes = split_idx['test'].numpy()
    np.random.seed(42)
    sampled_nodes = np.random.choice(test_nodes, size=NUM_SAMPLE_NODES, replace=False)
    print(f"  ✓ Sampled nodes: {sampled_nodes}")
    
    # Test each explainer
    print(f"\n[3/3] Running distributed benchmark tests...")
    
    for explainer_name in ['heuchase', 'apxchase']:  # Skip gnnexplainer for now (slower)
        print(f"\n{'='*70}")
        print(f"Testing: {explainer_name} with {NUM_WORKERS} workers")
        print(f"{'='*70}")
        
        try:
            result = run_distributed_benchmark(
                data=data,
                model_path=MODEL_PATH,
                node_ids=sampled_nodes,
                explainer_name=explainer_name,
                num_workers=NUM_WORKERS,
                explainer_config=EXPLAINER_CONFIGS[explainer_name],
                num_hops=NUM_HOPS,
                device=DEVICE
            )
            
            print(f"\n✓ {explainer_name} completed successfully!")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Extraction time: {result['extraction_time']:.2f}s")
            print(f"  Parallel time: {result['parallel_time']:.2f}s")
            print(f"  Mean task runtime: {result['task_runtime_mean']:.3f}s")
            print(f"  Load balance ratio: {result['load_balance_ratio']:.3f}")
            
        except Exception as e:
            print(f"\n✗ {explainer_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Quick Test Completed!")
    print("=" * 70)
    print("\nIf all tests passed, you can run the full benchmark:")
    print("  sbatch run_ogbn_distributed_bench.slurm")
