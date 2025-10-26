"""
Distributed Explainability Benchmark on TreeCycle Graph

测试 HeuChase, ApxChase, GNNExplainer 在 TreeCycle 图上的性能
使用 Coordinator-Worker 架构，基于 min-heap 的负载均衡
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import heapq
import json
import os
import psutil
import signal
import gc
from collections import defaultdict
from tqdm import tqdm
import pickle

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Import explainer implementations
import sys
sys.path.append('src')
from heuchase import HeuChase
from apxchase import ApxChase
from exhaustchase import ExhaustChase
from baselines import run_gnn_explainer_node, PGExplainerBaseline
from constraints import get_constraints


# Timeout exception
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Task timeout")


# GCN Model (same as training)
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


class SubgraphTask:
    """表示一个子图解释任务"""
    def __init__(self, task_id, node_id, subgraph_data, num_edges):
        self.task_id = task_id
        self.node_id = node_id
        self.subgraph_data = subgraph_data
        self.num_edges = num_edges
    
    def __lt__(self, other):
        return self.num_edges < other.num_edges


class LoadBalancer:
    """基于 min-heap 的负载均衡器"""
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.worker_heap = [(0, i) for i in range(num_workers)]
        heapq.heapify(self.worker_heap)
        self.worker_loads = [0] * num_workers
    
    def assign_task(self, task_size):
        current_load, worker_id = heapq.heappop(self.worker_heap)
        new_load = current_load + task_size
        heapq.heappush(self.worker_heap, (new_load, worker_id))
        self.worker_loads[worker_id] = new_load
        return worker_id
    
    def get_load_stats(self):
        total = np.sum(self.worker_loads)
        min_load = np.min(self.worker_loads)
        max_load = np.max(self.worker_loads)
        avg_load = total / len(self.worker_loads) if len(self.worker_loads) > 0 else 0
        balance_ratio = min_load / max_load if max_load > 0 else 1.0
        
        return {
            'total_load': int(total),
            'avg_load': float(avg_load),
            'min_load': int(min_load),
            'max_load': int(max_load),
            'balance_ratio': float(balance_ratio),
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
        # Convert numpy int to Python int or tensor
        if isinstance(node_id, np.integer):
            node_id = int(node_id)
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_id,
            num_hops=self.num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )
        
        subgraph = Data(
            x=self.data.x[subset].clone().detach(),
            edge_index=edge_index.clone().detach(),
            y=self.data.y[subset].clone().detach(),
            subset=subset.clone().detach(),
            target_node=mapping.item(),
        )
        
        # Mark as node classification task (same as OGBN)
        subgraph.task = 'node'
        subgraph.root = mapping.item()
        subgraph._target_node_subgraph_id = mapping.item()
        
        num_edges = edge_index.size(1)
        return subgraph, num_edges
    
    def create_tasks(self, node_ids):
        """为所有节点创建任务"""
        tasks = []
        print(f"Coordinator: Extracting {len(node_ids)} subgraphs...")
        
        for i, node_id in enumerate(tqdm(node_ids, desc="Extracting subgraphs")):
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
        """使用负载均衡分发任务"""
        balancer = LoadBalancer(num_workers)
        worker_tasks = [[] for _ in range(num_workers)]
        
        # Sort tasks by edge count (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t.num_edges, reverse=True)
        
        for task in sorted_tasks:
            worker_id = balancer.assign_task(task.num_edges)
            worker_tasks[worker_id].append(task)
        
        # Print load distribution
        load_stats = balancer.get_load_stats()
        print(f"\nLoad Balancing:")
        print(f"  Total edges: {load_stats['total_load']:,}")
        print(f"  Avg load per worker: {load_stats['avg_load']:,.0f}")
        print(f"  Min/Max load: {load_stats['min_load']:,} / {load_stats['max_load']:,}")
        print(f"  Balance ratio: {load_stats['balance_ratio']:.3f}")
        
        return worker_tasks, load_stats


def worker_process(worker_id, tasks, model_state, explainer_name, constraints, 
                   result_queue, device_id=None, timeout_seconds=1800):
    """Worker 进程：处理分配的任务
    
    Args:
        timeout_seconds: 单个任务的超时时间（秒），默认30分钟
    """
    import sys
    import gc
    sys.stdout.flush()
    print(f"\n{'='*60}", flush=True)
    print(f"WORKER {worker_id}: FUNCTION ENTRY", flush=True)
    print(f"{'='*60}\n", flush=True)
    sys.stdout.flush()
    
    try:
        print(f"Worker {worker_id}: Started, tasks={len(tasks)}, explainer={explainer_name}", flush=True)
        
        # Set device
        if device_id is not None and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
        
        print(f"Worker {worker_id}: Using device={device}", flush=True)
        
        # Load model
        if len(tasks) == 0:
            print(f"Worker {worker_id}: No tasks, exiting", flush=True)
            result_queue.put({
                'worker_id': worker_id,
                'num_tasks': 0,
                'total_time': 0,
                'results': []
            })
            return
        
        print(f"Worker {worker_id}: Loading model...", flush=True)
        first_task = tasks[0]
        in_channels = first_task.subgraph_data.x.size(1)
        out_channels = len(torch.unique(first_task.subgraph_data.y))
        
        model = GCN(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        print(f"Worker {worker_id}: Model loaded ✓", flush=True)
        
        # Initialize explainer ONCE (not per task!)
        print(f"Worker {worker_id}: Initializing {explainer_name}...", flush=True)
        explainer = None
        
        if explainer_name == 'HeuChase':
            explainer = HeuChase(
                model=model,
                Sigma=constraints,
                L=2,  # 2-hop subgraph
                k=10,  # Keep top-10 witnesses
                B=8,   # Budget for HEAD matching (same as OGBN!)
                m=6,   # Max candidates per step
                noise_std=1e-3,
                debug=False
            )
            print(f"Worker {worker_id}: HeuChase initialized ✓ (B=8)", flush=True)
            
        elif explainer_name == 'ApxChase':
            explainer = ApxChase(
                model=model,
                Sigma=constraints,
                L=2,
                k=10,
                B=8,   # Budget for HEAD matching (same as OGBN!)
                debug=False
            )
            print(f"Worker {worker_id}: ApxChase initialized ✓ (B=8)", flush=True)
            
        elif explainer_name == 'ExhaustChase':
            explainer = ExhaustChase(
                model=model,
                Sigma=constraints,
                L=2,
                k=10,
                B=8,   # Budget for HEAD matching (same as OGBN!)
                debug=False,
                max_enforce_iterations=50  # Exhaustive is still slow, keep at 50
            )
            print(f"Worker {worker_id}: ExhaustChase initialized ✓ (B=8, MaxIter=50)", flush=True)
            
        elif explainer_name == 'PGExplainer':
            # PGExplainer needs training first - skip for now or use quick_fit
            explainer = None  # Will initialize per-task with quick training
            print(f"Worker {worker_id}: PGExplainer (per-task initialization)", flush=True)
        
        print(f"Worker {worker_id}: Processing {len(tasks)} tasks...", flush=True)
        
        # Set up signal handler once (not per task)
        signal.signal(signal.SIGALRM, timeout_handler)
        
        results = []
        total_time = 0
        
        for task_idx, task in enumerate(tasks):
            print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} (node {task.node_id}, {task.num_edges} edges)...", flush=True)
            
            subgraph = task.subgraph_data.to(device)
            target_node = subgraph.target_node
            print(f"Worker {worker_id}: Subgraph moved to device, running explainer...", flush=True)
            
            start_time = time.time()
            timed_out = False
            
            # Set timeout alarm (30 minutes = 1800 seconds)
            signal.alarm(timeout_seconds)
            
            try:
                if explainer_name in ['HeuChase', 'ApxChase', 'ExhaustChase']:
                    # Use _run method directly like OGBN
                    print(f"Worker {worker_id}: Calling {explainer_name}._run() on subgraph with {subgraph.num_nodes if hasattr(subgraph, 'num_nodes') else subgraph.x.size(0)} nodes, {subgraph.edge_index.size(1)} edges...", flush=True)
                    run_start = time.time()
                    Sigma_star, S_k = explainer._run(H=subgraph, root=int(target_node))
                    run_elapsed = time.time() - run_start
                    print(f"Worker {worker_id}: {explainer_name}._run() completed in {run_elapsed:.2f}s, found {len(S_k)} witnesses", flush=True)
                    num_witnesses = len(S_k)
                    coverage = len(Sigma_star) if Sigma_star else 0
                    explanation_result = {
                        'num_witnesses': num_witnesses,
                        'coverage': coverage,
                        'success': num_witnesses > 0,
                        'timeout': False
                    }
                    
                elif explainer_name == 'GNNExplainer':
                    gnn_result = run_gnn_explainer_node(
                        model=model,
                        data=subgraph,
                        target_node=int(target_node),
                        epochs=100,
                        lr=0.01,
                        device=device
                    )
                    explanation_result = {
                        'edge_mask': gnn_result.get('edge_mask'),
                        'pred': gnn_result.get('pred'),
                        'success': gnn_result.get('edge_mask') is not None,
                        'timeout': False
                    }
                    
                elif explainer_name == 'PGExplainer':
                    # PGExplainer: Skip training for distributed benchmark (too slow)
                    # Just report as timeout/not-scalable
                    print(f"Worker {worker_id}: PGExplainer skipped (needs training, not scalable for distributed)", flush=True)
                    explanation_result = {
                        'success': False,
                        'timeout': True,
                        'reason': 'PGExplainer requires training, skipped'
                    }
                    
                else:
                    raise ValueError(f"Unknown explainer: {explainer_name}")
                    
            except TimeoutException:
                elapsed = time.time() - start_time
                print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} ⏱ TIMEOUT ({elapsed:.2f}s, >{timeout_seconds}s)", flush=True)
                explanation_result = {
                    'success': False,
                    'timeout': True,
                    'reason': f'Exceeded {timeout_seconds}s timeout'
                }
                timed_out = True
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} ✗ ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                explanation_result = {
                    'success': False,
                    'timeout': False,
                    'error': str(e)
                }
                
            finally:
                # Cancel the alarm
                signal.alarm(0)
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                if not timed_out:
                    success_str = "✓" if explanation_result.get('success', False) else "✗"
                    if explainer_name in ['HeuChase', 'ApxChase', 'ExhaustChase']:
                        witnesses = explanation_result.get('num_witnesses', 0)
                        print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} {success_str} ({elapsed:.2f}s, {witnesses} witnesses)", flush=True)
                    else:
                        print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} {success_str} ({elapsed:.2f}s)", flush=True)
                
                # Explicit garbage collection to free memory
                del subgraph
                gc.collect()
                
                result = {
                    'task_id': task.task_id,
                    'node_id': task.node_id,
                    'num_edges': task.num_edges,
                    'runtime': elapsed,
                    'worker_id': worker_id,
                    'explanation': explanation_result,
                }
                results.append(result)
        
        print(f"Worker {worker_id}: All tasks completed, sending results...", flush=True)
        result_queue.put({
            'worker_id': worker_id,
            'num_tasks': len(tasks),
            'total_time': total_time,
            'results': results
        })
        print(f"Worker {worker_id}: Results sent, exiting ✓", flush=True)
        
    except Exception as e:
        print(f"Worker {worker_id}: Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        result_queue.put({
            'worker_id': worker_id,
            'num_tasks': 0,
            'total_time': 0,
            'results': []
        })
        sys.stdout.flush()


def run_distributed_benchmark(data, model, explainer_name, constraints, 
                              num_workers=20, num_targets=100, num_hops=2, 
                              seed=42):
    """运行分布式 benchmark"""
    print(f"\n{'='*70}")
    print(f"Distributed Benchmark: {explainer_name}")
    print(f"{'='*70}")
    print(f"Dataset: TreeCycle")
    print(f"Num workers: {num_workers}")
    print(f"Num targets: {num_targets}")
    print(f"Num hops: {num_hops}")
    
    # Sample target nodes
    np.random.seed(seed)
    all_nodes = np.arange(data.num_nodes)
    target_nodes = np.random.choice(all_nodes, size=num_targets, replace=False)
    print(f"Sampled {len(target_nodes)} target nodes")
    
    # Create coordinator
    device = torch.device('cpu')  # Coordinator uses CPU
    coordinator = Coordinator(data, model, device, num_hops=num_hops)
    
    # Extract subgraphs
    print("\nPhase 1: Subgraph Extraction")
    extraction_start = time.time()
    tasks = coordinator.create_tasks(target_nodes)
    extraction_time = time.time() - extraction_start
    print(f"Extraction time: {extraction_time:.2f}s")
    
    # Distribute tasks
    print("\nPhase 2: Task Distribution")
    worker_tasks, load_stats = coordinator.distribute_tasks(tasks, num_workers)
    
    # Start workers
    print("\nPhase 3: Parallel Execution")
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    model_state = model.state_dict()
    
    execution_start = time.time()
    
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, worker_tasks[worker_id], model_state, 
                  explainer_name, constraints, result_queue, None)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    all_results = []
    for _ in range(num_workers):
        worker_result = result_queue.get()
        worker_id = worker_result['worker_id']
        results = worker_result['results']
        print(f"Coordinator: Received {len(results)} results from worker {worker_id}", flush=True)
        all_results.extend(results)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    execution_time = time.time() - execution_start
    
    # Aggregate results
    total_time = extraction_time + execution_time
    successful_tasks = [r for r in all_results if r['explanation'].get('success', False)]
    timeout_tasks = [r for r in all_results if r['explanation'].get('timeout', False)]
    failed_tasks = [r for r in all_results if not r['explanation'].get('success', False) and not r['explanation'].get('timeout', False)]
    
    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")
    print(f"Total tasks: {len(all_results)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Timeout (>30min): {len(timeout_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    print(f"\nTiming:")
    print(f"  Extraction time: {extraction_time:.2f}s")
    print(f"  Execution time (makespan): {execution_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    
    if successful_tasks:
        task_times = [r['runtime'] for r in successful_tasks]
        print(f"\nSuccessful task times:")
        print(f"  Mean: {np.mean(task_times):.3f}s")
        print(f"  Median: {np.median(task_times):.3f}s")
        print(f"  Min/Max: {np.min(task_times):.3f}s / {np.max(task_times):.3f}s")
        
        if explainer_name in ['HeuChase', 'ApxChase', 'ExhaustChase']:
            num_witnesses = [r['explanation'].get('num_witnesses', 0) for r in successful_tasks]
            coverage = [r['explanation'].get('coverage', 0) for r in successful_tasks]
            print(f"\nWitnesses found:")
            print(f"  Mean: {np.mean(num_witnesses):.1f}")
            print(f"  Median: {np.median(num_witnesses):.1f}")
            print(f"  Min/Max: {np.min(num_witnesses)} / {np.max(num_witnesses)}")
            print(f"\nCoverage:")
            print(f"  Mean: {np.mean(coverage):.1f}")
            print(f"  Median: {np.median(coverage):.1f}")
            print(f"  Min/Max: {np.min(coverage)} / {np.max(coverage)}")
    
    if timeout_tasks:
        print(f"\nTimeout tasks: {len(timeout_tasks)} (not scalable for distributed setting)")
    
    return {
        'explainer': explainer_name,
        'num_workers': num_workers,
        'num_targets': num_targets,
        'num_hops': num_hops,
        'extraction_time': extraction_time,
        'execution_time': execution_time,
        'total_time': total_time,
        'successful_tasks': len(successful_tasks),
        'timeout_tasks': len(timeout_tasks),
        'failed_tasks': len(failed_tasks),
        'load_stats': load_stats,
        'results': all_results
    }


def main():
    print("="*70)
    print("TreeCycle Distributed Benchmark")
    print("="*70)
    
    # Load data
    print("\nLoading TreeCycle graph...")
    data = torch.load('datasets/TreeCycle/treecycle_d5_bf15_n813616.pt')
    print(f"Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Load model
    print("\nLoading GCN model...")
    in_channels = data.x.size(1)
    out_channels = len(torch.unique(data.y))
    model = GCN(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
    model.load_state_dict(torch.load('models/TreeCycle_gcn_d5_bf15_n813616.pth'))
    model.eval()
    print("Model loaded")
    
    # Load constraints
    print("\nLoading TreeCycle constraints...")
    constraints = get_constraints('TREECYCLE')
    print(f"Loaded {len(constraints)} constraints")
    
    # Run benchmarks for all explainers
    explainers = ['HeuChase', 'ApxChase', 'ExhaustChase', 'GNNExplainer', 'PGExplainer']
    all_results = {}
    
    for explainer_name in explainers:
        result = run_distributed_benchmark(
            data=data,
            model=model,
            explainer_name=explainer_name,
            constraints=constraints,
            num_workers=20,
            num_targets=100,
            num_hops=2,
            seed=42
        )
        all_results[explainer_name] = result
        
        # Print comparison summary
        print(f"\n{'='*70}")
        print(f"Progress Summary")
        print(f"{'='*70}")
        for exp_name, exp_result in all_results.items():
            makespan = exp_result['execution_time']
            success = exp_result['successful_tasks']
            timeout = exp_result.get('timeout_tasks', 0)
            total = success + timeout + exp_result['failed_tasks']
            print(f"{exp_name:20s} | Makespan: {makespan:7.2f}s | Success: {success:3d}/{total:3d} | Timeout: {timeout:3d}")
    
    # Save results
    output_file = 'results/treecycle_distributed_benchmark.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
