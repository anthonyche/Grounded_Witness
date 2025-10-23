"""
Distributed Explainability Benchmark on OGBN-Papers100M

测试 HeuIChase, ApxIChase, GNNExplainer 在不同并行度下的性能
使用 Coordinator-Worker 架构，基于 min-heap 的负载均衡

Architecture:
- Coordinator: 提取 L-hop 子图，分发任务
- Workers: 并行运行解释算法
- Load Balancing: Min-heap based on subgraph edge count
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import heapq
import json
import os
import psutil  # For memory monitoring
from collections import defaultdict
from tqdm import tqdm
import pickle

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from ogb.nodeproppred import PygNodePropPredDataset

# Import your explainer implementations
import sys
sys.path.append('src')
from model import *
from Train_OGBN_HPC_MiniBatch import GCN_2_OGBN
from heuchase import HeuChase
from apxchase import ApxChase
from baselines import run_gnn_explainer_node
from constraints import get_constraints


class SubgraphTask:
    """表示一个子图解释任务"""
    def __init__(self, task_id, node_id, subgraph_data, num_edges):
        self.task_id = task_id
        self.node_id = node_id
        self.subgraph_data = subgraph_data
        self.num_edges = num_edges
    
    def __lt__(self, other):
        # For min-heap: smaller edge count = higher priority
        return self.num_edges < other.num_edges


class LoadBalancer:
    """基于 min-heap 的负载均衡器"""
    def __init__(self, num_workers):
        self.num_workers = num_workers
        # Heap: (current_load, worker_id)
        self.worker_heap = [(0, i) for i in range(num_workers)]
        heapq.heapify(self.worker_heap)
        self.worker_loads = [0] * num_workers
    
    def assign_task(self, task_size):
        """分配任务到负载最小的 worker"""
        current_load, worker_id = heapq.heappop(self.worker_heap)
        new_load = current_load + task_size
        heapq.heappush(self.worker_heap, (new_load, worker_id))
        self.worker_loads[worker_id] = new_load
        return worker_id
    
    def get_load_stats(self):
        """获取负载统计"""
        return {
            'mean': np.mean(self.worker_loads),
            'std': np.std(self.worker_loads),
            'min': np.min(self.worker_loads),
            'max': np.max(self.worker_loads),
            'loads': self.worker_loads.copy()
        }


class Coordinator:
    """Coordinator: 提取子图并分发任务"""
    def __init__(self, data, model, device, num_hops=2):
        self.data = data
        self.model = model
        self.device = device
        self.num_hops = num_hops
        
    def extract_subgraph(self, node_id):
        """提取节点的 L-hop 邻居子图"""
        try:
            # Extract k-hop subgraph
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=node_id,
                num_hops=self.num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes,
            )
            
            # Create subgraph data
            subgraph = Data(
                x=self.data.x[subset].clone().detach(),  # Clone to avoid memory sharing
                edge_index=edge_index.clone().detach(),
                y=self.data.y[subset].clone().detach(),
                subset=subset.clone().detach(),
                target_node=mapping.item(),  # Target node in subgraph
            )
            
            num_edges = edge_index.size(1)
            return subgraph, num_edges
            
        except Exception as e:
            print(f"\nERROR extracting subgraph for node {node_id}: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_tasks(self, node_ids):
        """为所有节点创建任务"""
        tasks = []
        print(f"Coordinator: Extracting {len(node_ids)} subgraphs...")
        
        for i, node_id in enumerate(tqdm(node_ids, desc="Extracting subgraphs")):
            try:
                subgraph, num_edges = self.extract_subgraph(node_id)
            except Exception as e:
                print(f"\nFailed to extract subgraph for node {node_id} (task {i})")
                raise
            subgraph, num_edges = self.extract_subgraph(node_id)
            task = SubgraphTask(
                task_id=i,
                node_id=node_id,
                subgraph_data=subgraph,
                num_edges=num_edges
            )
            tasks.append(task)
        
        print(f"Coordinator: Created {len(tasks)} tasks")
        print(f"  Subgraph sizes (edges): min={min(t.num_edges for t in tasks)}, "
              f"max={max(t.num_edges for t in tasks)}, "
              f"mean={np.mean([t.num_edges for t in tasks]):.1f}")
        
        return tasks
    
    def distribute_tasks(self, tasks, num_workers):
        """使用 min-heap 负载均衡分配任务"""
        load_balancer = LoadBalancer(num_workers)
        task_assignments = [[] for _ in range(num_workers)]
        
        # Sort tasks by size (descending) for better load balancing
        sorted_tasks = sorted(tasks, key=lambda t: t.num_edges, reverse=True)
        
        print(f"Coordinator: Distributing tasks to {num_workers} workers...")
        for task in sorted_tasks:
            worker_id = load_balancer.assign_task(task.num_edges)
            task_assignments[worker_id].append(task)
        
        # Print load statistics
        load_stats = load_balancer.get_load_stats()
        print(f"Coordinator: Load distribution statistics:")
        print(f"  Mean load: {load_stats['mean']:.1f} edges")
        print(f"  Std dev: {load_stats['std']:.1f}")
        print(f"  Min/Max: {load_stats['min']}/{load_stats['max']}")
        print(f"  Load balance ratio: {load_stats['min']/load_stats['max']:.2f}")
        
        return task_assignments, load_stats


def worker_process(worker_id, tasks, model_state, explainer_name, explainer_config, device, result_queue):
    """Worker 进程：运行解释算法"""
    try:
        # Load model
        model = GCN_2_OGBN(
            input_dim=128,
            hidden_dim=model_state['hidden_dim'],
            output_dim=172,
            dropout=0.5
        ).to(device)
        model.load_state_dict(model_state['model_state_dict'])
        model.eval()
        
        # Initialize explainer
        if explainer_name == 'heuchase':
            # HeuChase: Edmonds-based witness generation
            explainer = HeuChase(
                model=model,
                Sigma=explainer_config.get('Sigma', None),  # Constraints (can be None for node classification)
                L=explainer_config.get('L', 2),  # L-hop subgraph
                k=explainer_config.get('k', 10),  # window size
                B=explainer_config.get('B', 5),  # budget
                m=explainer_config.get('m', 6),  # number of Edmonds candidates
                noise_std=explainer_config.get('noise_std', 1e-3),
                debug=False,
            )
        elif explainer_name == 'apxchase':
            # ApxChase: Streaming edge-insertion chase
            explainer = ApxChase(
                model=model,
                Sigma=explainer_config.get('Sigma', None),
                L=explainer_config.get('L', 2),
                k=explainer_config.get('k', 10),
                B=explainer_config.get('B', 5),
                debug=False,
            )
        elif explainer_name == 'gnnexplainer':
            # GNNExplainer: PyG baseline
            explainer = None  # Will use run_gnn_explainer_node function
        else:
            raise ValueError(f"Unknown explainer: {explainer_name}")
        
        results = []
        total_time = 0
        
        print(f"Worker {worker_id}: Processing {len(tasks)} tasks...")
        
        for task in tasks:
            start_time = time.time()
            
            # Move subgraph to device
            subgraph = task.subgraph_data.to(device)
            target_node = subgraph.target_node  # Target node in subgraph coordinates
            
            try:
                # Run explanation based on explainer type
                if explainer_name in ['heuchase', 'apxchase']:
                    # For chase-based explainers: call _run method
                    # root = target node for node classification
                    Sigma_star, S_k = explainer._run(H=subgraph, root=int(target_node))
                    
                    # Extract explanation result
                    num_witnesses = len(S_k)
                    coverage = len(Sigma_star) if Sigma_star else 0
                    explanation_result = {
                        'num_witnesses': num_witnesses,
                        'coverage': coverage,
                        'success': num_witnesses > 0
                    }
                    
                elif explainer_name == 'gnnexplainer':
                    # For GNNExplainer: use run_gnn_explainer_node
                    gnn_result = run_gnn_explainer_node(
                        model=model,
                        data=subgraph,
                        target_node=int(target_node),
                        epochs=explainer_config.get('epochs', 100),
                        lr=explainer_config.get('lr', 0.01),
                        device=device,
                    )
                    
                    explanation_result = {
                        'edge_mask': gnn_result['edge_mask'],
                        'pred': gnn_result['pred'],
                        'success': gnn_result['edge_mask'] is not None
                    }
                
            except Exception as e:
                print(f"Worker {worker_id}: Error explaining node {task.node_id}: {e}")
                explanation_result = {
                    'success': False,
                    'error': str(e)
                }
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            result = {
                'task_id': task.task_id,
                'node_id': task.node_id,
                'num_edges': task.num_edges,
                'runtime': elapsed,
                'worker_id': worker_id,
                'explanation': explanation_result,
            }
            results.append(result)
        
        # Send results back
        result_queue.put({
            'worker_id': worker_id,
            'num_tasks': len(tasks),
            'total_time': total_time,
            'results': results,
        })
        
        print(f"Worker {worker_id}: Completed {len(tasks)} tasks in {total_time:.2f}s")
        
    except Exception as e:
        print(f"Worker {worker_id}: Error - {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'worker_id': worker_id,
            'error': str(e),
        })


def run_distributed_benchmark(
    data,
    model_path,
    node_ids,
    explainer_name,
    num_workers,
    explainer_config=None,
    num_hops=2,
    device='cpu'
):
    """运行分布式基准测试"""
    
    if explainer_config is None:
        explainer_config = {}
    
    print("="*70)
    print(f"Starting Distributed Benchmark")
    print(f"  Explainer: {explainer_name}")
    print(f"  Num workers: {num_workers}")
    print(f"  Num nodes: {len(node_ids)}")
    print(f"  Num hops: {num_hops}")
    print(f"  Explainer config: {explainer_config}")
    print("="*70)
    
    try:
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        model = GCN_2_OGBN(
            input_dim=128,
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=172,
            dropout=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  Model loaded successfully (hidden_dim={checkpoint['hidden_dim']})")
        
        # Print memory before subgraph extraction
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3
        print(f"  Memory before subgraph extraction: {mem_before:.2f} GB")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Phase 1: Coordinator extracts subgraphs and creates tasks
    try:
        coordinator = Coordinator(data, model, device, num_hops=num_hops)
    except Exception as e:
        print(f"ERROR creating coordinator: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        start_time = time.time()
        tasks = coordinator.create_tasks(node_ids)
        extraction_time = time.time() - start_time
        
        # Print memory after subgraph extraction
        mem_after = process.memory_info().rss / 1024**3
        print(f"  Memory after subgraph extraction: {mem_after:.2f} GB (+{mem_after - mem_before:.2f} GB)")
        
    except Exception as e:
        print(f"ERROR creating tasks (extracting subgraphs): {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Phase 2: Distribute tasks with load balancing
    task_assignments, load_stats = coordinator.distribute_tasks(tasks, num_workers)
    
    # Phase 3: Workers process tasks in parallel
    print(f"\nStarting {num_workers} worker processes...")
    start_time = time.time()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for worker_id in range(num_workers):
        worker_tasks = task_assignments[worker_id]
        p = mp.Process(
            target=worker_process,
            args=(worker_id, worker_tasks, checkpoint, explainer_name, explainer_config, device, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    worker_results = []
    for _ in range(num_workers):
        result = result_queue.get()
        worker_results.append(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_parallel_time = time.time() - start_time
    
    # Phase 4: Aggregate results
    all_task_results = []
    for worker_result in worker_results:
        if 'error' not in worker_result:
            all_task_results.extend(worker_result['results'])
    
    # Compute statistics
    task_runtimes = [r['runtime'] for r in all_task_results]
    worker_times = [r['total_time'] for r in worker_results if 'error' not in r]
    
    benchmark_result = {
        'explainer': explainer_name,
        'num_workers': num_workers,
        'num_nodes': len(node_ids),
        'num_hops': num_hops,
        
        # Timing
        'extraction_time': extraction_time,
        'parallel_time': total_parallel_time,
        'total_time': extraction_time + total_parallel_time,
        
        # Task statistics
        'task_runtime_mean': np.mean(task_runtimes),
        'task_runtime_std': np.std(task_runtimes),
        'task_runtime_min': np.min(task_runtimes),
        'task_runtime_max': np.max(task_runtimes),
        
        # Worker statistics
        'worker_time_mean': np.mean(worker_times),
        'worker_time_std': np.std(worker_times),
        'worker_time_min': np.min(worker_times),
        'worker_time_max': np.max(worker_times),
        
        # Load balancing
        'load_balance_ratio': load_stats['min'] / load_stats['max'],
        'load_stats': load_stats,
        
        # Detailed results
        'task_results': all_task_results,
        'worker_results': worker_results,
    }
    
    print("\n" + "="*70)
    print("Benchmark Results")
    print("="*70)
    print(f"Total time: {benchmark_result['total_time']:.2f}s")
    print(f"  - Extraction: {extraction_time:.2f}s")
    print(f"  - Parallel execution: {total_parallel_time:.2f}s")
    print(f"Task runtime: {benchmark_result['task_runtime_mean']:.3f}s ± {benchmark_result['task_runtime_std']:.3f}s")
    print(f"Worker time: {benchmark_result['worker_time_mean']:.2f}s ± {benchmark_result['worker_time_std']:.2f}s")
    print(f"Load balance ratio: {benchmark_result['load_balance_ratio']:.3f}")
    print(f"Speedup: {benchmark_result['worker_time_mean'] / total_parallel_time:.2f}x")
    print("="*70)
    
    return benchmark_result


def main():
    """主函数：运行完整实验"""
    
    import argparse
    import yaml
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OGBN-Papers100M Distributed Benchmark')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--explainers', nargs='+', default=None, help='List of explainers to test (heuchase, apxchase, gnnexplainer)')
    parser.add_argument('--workers', nargs='+', type=int, default=None, help='List of worker counts to test')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of nodes to sample')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    args = parser.parse_args()
    
    # Load config file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Using default configuration")
        config = {}
    
    # Configuration with priority: CLI args > config file > defaults
    NUM_SAMPLE_NODES = args.num_nodes if args.num_nodes is not None else config.get('num_target_nodes', 100)
    NUM_HOPS = config.get('L', 2)
    NUM_WORKERS_LIST = args.workers if args.workers is not None else config.get('num_workers', [2, 4, 6, 8, 10])
    
    # Determine explainers to test
    if args.explainers is not None:
        EXPLAINERS = args.explainers
    else:
        # Try to infer from config exp_name
        exp_name = config.get('exp_name', 'heuchase_ogbn')
        if 'heuchase' in exp_name.lower():
            EXPLAINERS = ['heuchase', 'apxchase', 'gnnexplainer']
        elif 'apxchase' in exp_name.lower():
            EXPLAINERS = ['apxchase', 'gnnexplainer']
        elif 'gnnexplainer' in exp_name.lower():
            EXPLAINERS = ['gnnexplainer']
        else:
            # Default: test all
            EXPLAINERS = ['heuchase', 'apxchase', 'gnnexplainer']
    
    MODEL_PATH = args.model_path if args.model_path is not None else 'models/OGBN_Papers100M_epoch_20.pth'
    DEVICE = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')  # Get device from config

    
    print("="*70)
    print("Distributed Benchmark Configuration")
    print("="*70)
    print(f"  Config file: {args.config}")
    print(f"  Explainers: {EXPLAINERS}")
    print(f"  Worker counts: {NUM_WORKERS_LIST}")
    print(f"  Sample nodes: {NUM_SAMPLE_NODES}")
    print(f"  Num hops: {NUM_HOPS}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    print("="*70)
    
    print("="*70)
    
    # Load constraints for OGBN-Papers100M
    print("\nLoading constraints for OGBN-Papers100M...")
    CONSTRAINTS = get_constraints('OGBN-PAPERS100M')
    print(f"  Loaded {len(CONSTRAINTS)} constraints:")
    for i, tgd in enumerate(CONSTRAINTS, 1):
        print(f"    {i}. {tgd['name']}")
    
    # Explainer-specific configurations from config file
    EXPLAINER_CONFIGS = {
        'heuchase': {
            'Sigma': CONSTRAINTS,  # Use OGBN-Papers100M constraints
            'L': NUM_HOPS,
            'k': config.get('k', 10),
            'B': config.get('Budget', 5),
            'm': config.get('heuchase_m', 6),
            'noise_std': config.get('heuchase_noise_std', 1e-3),
        },
        'apxchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,
            'k': config.get('k', 10),
            'B': config.get('Budget', 5),
            'alpha': config.get('alpha', 1.0),
            'beta': config.get('beta', 0.0),
            'gamma': config.get('gamma', 1.0),
        },
        'gnnexplainer': {
            'epochs': 100,
            'lr': 0.01,
        }
    }
    
    print("\nLoading OGBN-Papers100M dataset...")
    try:
        dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='./datasets')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        
        print(f"  Dataset loaded successfully!")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.edge_index.size(1):,}")
        print(f"  Features shape: {data.x.shape}")
        print(f"  Labels shape: {data.y.shape}")
        
        # Print memory usage
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        print(f"  Current memory usage: {mem_gb:.2f} GB")
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Sample test nodes
    test_nodes = split_idx['test'].numpy()
    np.random.seed(42)
    sampled_nodes = np.random.choice(test_nodes, size=NUM_SAMPLE_NODES, replace=False)
    
    print(f"\nSampled {len(sampled_nodes)} test nodes for explanation")
    
    # Run benchmarks
    all_results = []
    
    for explainer_name in EXPLAINERS:
        for num_workers in NUM_WORKERS_LIST:
            print(f"\n{'='*70}")
            print(f"Benchmark: {explainer_name} with {num_workers} workers")
            print(f"{'='*70}\n")
            
            result = run_distributed_benchmark(
                data=data,
                model_path=MODEL_PATH,
                node_ids=sampled_nodes,
                explainer_name=explainer_name,
                num_workers=num_workers,
                explainer_config=EXPLAINER_CONFIGS[explainer_name],
                num_hops=NUM_HOPS,
                device=DEVICE
            )
            
            all_results.append(result)
            
            # Save intermediate results
            os.makedirs('results/ogbn_distributed', exist_ok=True)
            result_file = f'results/ogbn_distributed/{explainer_name}_workers{num_workers}.json'
            with open(result_file, 'w') as f:
                # Remove detailed results for JSON
                summary = {k: v for k, v in result.items() if k not in ['task_results', 'worker_results']}
                json.dump(summary, f, indent=2)
            
            print(f"Saved results to {result_file}")
    
    # Save complete results
    with open('results/ogbn_distributed/complete_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n" + "="*70)
    print("All benchmarks completed!")
    print("="*70)
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Explainer':<15} {'Workers':<10} {'Total Time (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for result in all_results:
        speedup = result['worker_time_mean'] / result['parallel_time']
        print(f"{result['explainer']:<15} {result['num_workers']:<10} "
              f"{result['total_time']:<15.2f} {speedup:<10.2f}x")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
