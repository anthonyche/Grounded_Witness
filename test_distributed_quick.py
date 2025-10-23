"""
快速测试分布式基准 - 只测试 5 个节点，2 个 workers
验证完整的 Coordinator-Worker 架构
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import sys
import os
import yaml

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
    
    # Load config file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"\n[0/4] Loaded config from: config.yaml")
        print(f"  exp_name: {config.get('exp_name', 'N/A')}")
        print(f"  data_name: {config.get('data_name', 'N/A')}")
    except Exception as e:
        print(f"\n[0/4] Warning: Could not load config: {e}")
        config = {}
    
    # Configuration (CLI overrides config)
    NUM_SAMPLE_NODES = 5  # Very small for quick test
    NUM_HOPS = config.get('L', 2)
    NUM_WORKERS = 2
    MODEL_PATH = 'models/OGBN_Papers100M_epoch_20.pth'
    DEVICE = 'cpu'
    
    # Load constraints
    print(f"\n[1/4] Loading constraints for OGBN-Papers100M...")
    CONSTRAINTS = get_constraints('OGBN-PAPERS100M')
    print(f"  ✓ Loaded {len(CONSTRAINTS)} constraints:")
    for tgd in CONSTRAINTS:
        print(f"    - {tgd['name']}")
    
    # Explainer configurations from config file
    EXPLAINER_CONFIGS = {
        'heuchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,
            'k': config.get('k', 5),
            'B': config.get('Budget', 3),
            'm': config.get('heuchase_m', 3),  # Fewer candidates for quick test
            'noise_std': config.get('heuchase_noise_std', 1e-3),
        },
        'apxchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,
            'k': config.get('k', 5),
            'B': config.get('Budget', 3),
            'alpha': config.get('alpha', 1.0),
            'beta': config.get('beta', 0.0),
            'gamma': config.get('gamma', 1.0),
        },
        'gnnexplainer': {
            'epochs': 50,  # Fewer epochs for quick test
            'lr': 0.01,
        }
    }
    
    print(f"\nExplainer configs from config.yaml:")
    print(f"  k={config.get('k', 5)}, Budget={config.get('Budget', 3)}")
    print(f"  heuchase_m={config.get('heuchase_m', 3)}, alpha={config.get('alpha', 1.0)}")
    
    print(f"  heuchase_m={config.get('heuchase_m', 3)}, alpha={config.get('alpha', 1.0)}")
    
    # Load dataset
    print(f"\n[2/4] Loading OGBN-Papers100M dataset...")
    try:
        dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='./datasets')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        print(f"  ✓ Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        exit(1)
    
    # Sample nodes
    print(f"\n[3/4] Sampling {NUM_SAMPLE_NODES} test nodes...")
    test_nodes = split_idx['test'].numpy()
    np.random.seed(42)
    sampled_nodes = np.random.choice(test_nodes, size=NUM_SAMPLE_NODES, replace=False)
    print(f"  ✓ Sampled nodes: {sampled_nodes}")
    
    # Test each explainer
    print(f"\n[4/4] Running distributed benchmark tests...")
    
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
