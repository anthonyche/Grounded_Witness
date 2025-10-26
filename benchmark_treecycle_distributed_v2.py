"""
Distributed Explainability Benchmark on TreeCycle Graph

基于 OGBN benchmark 的 TreeCycle 版本
测试 HeuChase, ApxChase, ExhaustChase, GNNExplainer, PGExplainer
使用 Coordinator-Worker 架构，基于 min-heap 的负载均衡
"""

import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
import numpy as np
import time
import heapq
import json
import os
import psutil
import gc
from collections import defaultdict
from tqdm import tqdm
import pickle

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

# Import constraints (needed for main)
import sys
sys.path.append('src')
from constraints import get_constraints

# DO NOT import explainers here! Import inside worker_process to avoid multiprocessing issues
# from heuchase import HeuChase  # ← REMOVED
# from apxchase import ApxChase  # ← REMOVED  
# from exhaustchase import ExhaustChase  # ← REMOVED
# from baselines import run_gnn_explainer_node, PGExplainerBaseline  # ← REMOVED


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
        """分配任务到负载最小的 worker"""
        current_load, worker_id = heapq.heappop(self.worker_heap)
        new_load = current_load + task_size
        heapq.heappush(self.worker_heap, (new_load, worker_id))
        self.worker_loads[worker_id] = new_load
        return worker_id
    
    def get_load_stats(self):
        """获取负载统计"""
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
            'mean': float(np.mean(self.worker_loads)),
            'std': float(np.std(self.worker_loads)),
            'min': int(min_load),
            'max': int(max_load),
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
            # Convert node_id to int
            if isinstance(node_id, np.integer):
                node_id = int(node_id)
            
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
                x=self.data.x[subset].clone().detach(),
                edge_index=edge_index.clone().detach(),
                y=self.data.y[subset].clone().detach(),
                subset=subset.clone().detach(),
                target_node=mapping.item(),
            )
            
            # Mark as node classification task
            subgraph.task = 'node'
            subgraph.root = mapping.item()
            subgraph._target_node_subgraph_id = mapping.item()
            
            num_edges = edge_index.size(1)
            return subgraph, num_edges
            
        except Exception as e:
            print(f"\nERROR extracting subgraph for node {node_id}: {e}")
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
        
        load_stats = load_balancer.get_load_stats()
        print(f"  Load stats: min={load_stats['min_load']}, "
              f"max={load_stats['max_load']}, "
              f"avg={load_stats['avg_load']:.1f}, "
              f"balance={load_stats['balance_ratio']:.2f}")
        
        return task_assignments, load_stats


def worker_process(worker_id, tasks, model_state, explainer_name, explainer_config, device, result_queue):
    """Worker 进程：运行解释算法 (exact copy of OGBN structure)"""
    import sys
    import os
    import psutil
    import gc
    
    sys.stdout.flush()
    print(f"\n{'='*60}", flush=True)
    print(f"WORKER {worker_id}: FUNCTION ENTRY (PID: {os.getpid()})", flush=True)
    print(f"{'='*60}\n", flush=True)
    sys.stdout.flush()
    
    torch.set_num_threads(1)
    
    # Monitor memory
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024**3
    print(f"Worker {worker_id}: Initial memory: {mem_start:.2f} GB", flush=True)
    
    try:
        print(f"Worker {worker_id}: Started, device={device}, tasks={len(tasks)}", flush=True)
        sys.stdout.flush()
        
        # Import model from train_Treecycle.py
        print(f"Worker {worker_id}: Importing GCN model class...", flush=True)
        sys.stdout.flush()
        
        # Define GCN model (same as train_Treecycle.py)
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
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
        
        print(f"Worker {worker_id}: Model class loaded ✓", flush=True)
        sys.stdout.flush()
        
        # Create model
        print(f"Worker {worker_id}: Creating model...")
        model = GCN(
            in_channels=model_state['in_channels'],
            hidden_channels=model_state['hidden_channels'],
            out_channels=model_state['out_channels']
        )
        print(f"Worker {worker_id}: Model created, moving to {device}...")
        model = model.to(device)
        print(f"Worker {worker_id}: Model on device, loading state dict...")
        model.load_state_dict(model_state['state_dict'])
        model.eval()
        
        print(f"Worker {worker_id}: Model loaded and ready")

        print(f"Worker {worker_id}: Initializing {explainer_name} explainer...")

        # Initialize explainer (import inside worker to avoid multiprocessing issues)
        if explainer_name == 'heuchase':
            print(f"Worker {worker_id}: Importing HeuChase...")
            from heuchase import HeuChase
            print(f"Worker {worker_id}: Creating HeuChase...")
            explainer = HeuChase(
                model=model,
                Sigma=explainer_config.get('Sigma', None),
                L=explainer_config.get('L', 2),
                k=explainer_config.get('k', 10),
                B=explainer_config.get('B', 8),
                m=explainer_config.get('m', 6),
                noise_std=explainer_config.get('noise_std', 1e-3),
                debug=False,
            )
            print(f"Worker {worker_id}: HeuChase initialized (B={explainer_config.get('B', 8)})")

        elif explainer_name == 'apxchase':
            print(f"Worker {worker_id}: Importing ApxChase...")
            from apxchase import ApxChase
            print(f"Worker {worker_id}: Creating ApxChase...")
            explainer = ApxChase(
                model=model,
                Sigma=explainer_config.get('Sigma', None),
                L=explainer_config.get('L', 2),
                k=explainer_config.get('k', 10),
                B=explainer_config.get('B', 8),
                debug=False,
            )
            print(f"Worker {worker_id}: ApxChase initialized (B={explainer_config.get('B', 8)})")

        elif explainer_name == 'exhaustchase':
            print(f"Worker {worker_id}: Importing ExhaustChase...")
            from exhaustchase import ExhaustChase
            print(f"Worker {worker_id}: Creating ExhaustChase...")
            explainer = ExhaustChase(
                model=model,
                Sigma=explainer_config.get('Sigma', None),
                L=explainer_config.get('L', 2),
                k=explainer_config.get('k', 10),
                B=explainer_config.get('B', 8),
                debug=False,
                max_enforce_iterations=explainer_config.get('max_enforce_iterations', 50),
            )
            print(f"Worker {worker_id}: ExhaustChase initialized (B={explainer_config.get('B', 8)})")

        elif explainer_name == 'gnnexplainer':
            print(f"Worker {worker_id}: GNNExplainer uses baseline function (no init needed)")
            explainer = None

        elif explainer_name == 'pgexplainer':
            print(f"Worker {worker_id}: PGExplainer (skip - needs training)")
            explainer = None
        else:
            raise ValueError(f"Unknown explainer: {explainer_name}")

        print(f"Worker {worker_id}: ✓ Explainer ready, starting {len(tasks)} tasks")
        
        results = []
        total_time = 0
        
        print(f"Worker {worker_id}: Processing {len(tasks)} tasks...")
        
        for task_idx, task in enumerate(tasks):
            start_time = time.time()
            
            print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} (node {task.node_id}, {task.num_edges} edges)...")
            
            # Simple: move subgraph to same device as model (like OGBN)
            subgraph = task.subgraph_data.to(device)
            target_node = subgraph.target_node
            
            try:
                if explainer_name in ['heuchase', 'apxchase', 'exhaustchase']:
                    # For chase-based explainers: call _run method
                    Sigma_star, S_k = explainer._run(H=subgraph, root=int(target_node))
                    
                    num_witnesses = len(S_k)
                    coverage = len(Sigma_star) if Sigma_star else 0
                    explanation_result = {
                        'num_witnesses': num_witnesses,
                        'coverage': coverage,
                        'success': num_witnesses > 0
                    }
                    
                elif explainer_name == 'gnnexplainer':
                    # Import inside try block
                    from baselines import run_gnn_explainer_node
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
                    
                elif explainer_name == 'pgexplainer':
                    # Skip PGExplainer (needs training)
                    explanation_result = {
                        'success': False,
                        'reason': 'PGExplainer requires training, skipped'
                    }
                    
            except Exception as e:
                print(f"Worker {worker_id}: Error explaining node {task.node_id}: {e}")
                explanation_result = {
                    'success': False,
                    'error': str(e)
                }
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Print task completion
            success_str = "✓" if explanation_result.get('success', False) else "✗"
            if explainer_name in ['heuchase', 'apxchase', 'exhaustchase']:
                witnesses = explanation_result.get('num_witnesses', 0)
                print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} {success_str} ({elapsed:.2f}s, {witnesses} witnesses)")
            else:
                print(f"Worker {worker_id}: Task {task_idx+1}/{len(tasks)} {success_str} ({elapsed:.2f}s)")
            
            # Explicit garbage collection after each task
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
        
    except Exception as e:
        print(f"Worker {worker_id}: FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        result_queue.put({
            'worker_id': worker_id,
            'num_tasks': 0,
            'total_time': 0,
            'results': [],
            'error': str(e)
        })


def run_distributed_benchmark(model_path, tasks, explainer_name, 
                              num_workers=20, explainer_config=None, device='cpu'):
    """运行分布式基准测试
    
    Args:
        model_path: 模型路径
        tasks: 预提取的子图任务列表（从缓存加载）
        explainer_name: 解释器名称
        num_workers: worker 数量
        explainer_config: 解释器配置
        device: 模型设备 (cuda/cpu)
    """
    
    if explainer_config is None:
        explainer_config = {}
    
    print("="*70)
    print(f"TreeCycle Distributed Benchmark")
    print(f"  Explainer: {explainer_name}")
    print(f"  Num workers: {num_workers}")
    print(f"  Num tasks: {len(tasks)}")
    print(f"  Explainer config: {explainer_config}")
    print("="*70)
    
    try:
        # Load model
        print("Loading TreeCycle model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model dimensions from first task
        sample_task = tasks[0]
        in_channels = sample_task.subgraph_data.x.size(1)
        out_channels = len(torch.unique(sample_task.subgraph_data.y))
        hidden_channels = 32  # From train_Treecycle.py
        
        model_state = {
            'state_dict': checkpoint,
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels
        }
        
        print(f"  Model loaded (in={in_channels}, hidden={hidden_channels}, out={out_channels})")
        
        # Memory info
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3
        print(f"  Memory before extraction: {mem_before:.2f} GB")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Phase 1: Task Distribution (tasks already extracted from cache)
    print(f"\nPhase 1: Task Distribution")
    balancer = LoadBalancer(num_workers)
    task_assignments = [[] for _ in range(num_workers)]
    
    for task in tasks:
        worker_id = balancer.assign_task(task.num_edges)
        task_assignments[worker_id].append(task)
    
    load_stats = balancer.get_load_stats()
    print(f"  Load stats: min={load_stats['min_load']}, max={load_stats['max_load']}, "
          f"avg={load_stats['avg_load']:.0f}, balance={load_stats['balance_ratio']:.2f}")
    
    # Phase 2: Parallel execution
    print(f"\nPhase 2: Parallel Execution")
    print(f"  Starting {num_workers} workers on device={device}...")
    
    # mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    execution_start = time.time()
    
    import random, time as _tm
    for worker_id in range(num_workers):
        _tm.sleep(random.uniform(0, 0.05))
        p = mp.Process(
            target=worker_process,
            args=(worker_id, task_assignments[worker_id], model_state,
                  explainer_name, explainer_config, device, result_queue)
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
    total_time = execution_time  # No extraction time (from cache)
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
    print(f"  Execution time (makespan): {execution_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  (Subgraph extraction time: loaded from cache)")
    
    if successful_tasks:
        task_times = [r['runtime'] for r in successful_tasks]
        print(f"\nSuccessful task times:")
        print(f"  Mean: {np.mean(task_times):.3f}s")
        print(f"  Median: {np.median(task_times):.3f}s")
        print(f"  Min/Max: {np.min(task_times):.3f}s / {np.max(task_times):.3f}s")
        
        if explainer_name in ['heuchase', 'apxchase', 'exhaustchase']:
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
        'num_targets': len(tasks),
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
    
    # Check GPU availability
    print(f"\nGPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Configuration
    DATA_PATH = 'datasets/TreeCycle/treecycle_d5_bf15_n813616.pt'
    MODEL_PATH = 'models/TreeCycle_gcn_d5_bf15_n813616.pth'
    CACHE_DIR = 'cache/treecycle'
    NUM_WORKERS = 20
    NUM_TARGETS = 100
    NUM_HOPS = 2
    
    # Device strategy: Simple, like OGBN - pure CPU
    DEVICE = 'cpu'
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE} (pure CPU, like OGBN)")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Target nodes: {NUM_TARGETS}")
    print(f"  Hops: {NUM_HOPS}")
    print(f"  Cache directory: {CACHE_DIR}")
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Generate cache filename based on configuration
    cache_filename = f'subgraphs_n{NUM_TARGETS}_h{NUM_HOPS}_seed42.pkl'
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"\n✓ Found cached subgraphs: {cache_path}")
        print(f"  Loading from cache (avoid redundant sampling)...")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        sampled_nodes = cached_data['sampled_nodes']
        tasks = cached_data['tasks']
        extraction_time = 0.0  # No extraction needed
        
        print(f"  Loaded {len(tasks)} cached tasks")
        print(f"  Sampled nodes: {sampled_nodes[:10]}... (showing first 10)")
    else:
        print(f"\n✗ No cache found, will extract subgraphs...")
        
        # Load data
        print("\nLoading TreeCycle graph...")
        data = torch.load(DATA_PATH)
        print(f"  Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        
        # Sample target nodes (固定 seed=42)
        print(f"\nSampling {NUM_TARGETS} target nodes (seed=42)...")
        np.random.seed(42)
        sampled_nodes = np.random.choice(data.num_nodes, size=NUM_TARGETS, replace=False)
        print(f"  Sampled nodes: {sampled_nodes[:10]}... (showing first 10)")
        
        # Extract subgraphs
        print("\nExtracting subgraphs...")
        dummy_model = None
        coordinator = Coordinator(data, dummy_model, 'cpu', num_hops=NUM_HOPS)
        start_time = time.time()
        tasks = coordinator.create_tasks(sampled_nodes)
        extraction_time = time.time() - start_time
        print(f"  Extraction time: {extraction_time:.2f}s")
        
        # Save to cache
        print(f"\nSaving subgraphs to cache: {cache_path}")
        cached_data = {
            'sampled_nodes': sampled_nodes,
            'tasks': tasks,
            'num_targets': NUM_TARGETS,
            'num_hops': NUM_HOPS,
            'seed': 42,
            'extraction_time': extraction_time
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"  ✓ Cache saved")
    
    # Load constraints
    print("\nLoading TreeCycle constraints...")
    CONSTRAINTS = get_constraints('TREECYCLE')
    print(f"  Loaded {len(CONSTRAINTS)} constraints")
    
    # Explainer configurations (aligned with config.yaml)
    EXPLAINER_CONFIGS = {
        'heuchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,          # 2 (from config.yaml)
            'k': 6,                 # window size (from config.yaml)
            'B': 8,                 # Budget (from config.yaml)
            'm': 20,                # heuchase_m (from config.yaml)
            'noise_std': 0.2,       # heuchase_noise_std (from config.yaml)
        },
        'apxchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,          # 2
            'k': 6,                 # window size
            'B': 8,                 # Budget
        },
        'exhaustchase': {
            'Sigma': CONSTRAINTS,
            'L': NUM_HOPS,          # 2
            'k': 6,                 # window size
            'B': 8,                 # Budget
            'max_enforce_iterations': 50,  # max_enforce_iterations (from config.yaml)
        },
        'gnnexplainer': {
            'epochs': 100,
            'lr': 0.01,
        },
        'pgexplainer': {
            # Skipped - needs training
        }
    }
    
    # Run benchmarks
    EXPLAINERS = ['heuchase', 'apxchase', 'exhaustchase', 'gnnexplainer', 'pgexplainer']
    all_results = {}
    
    for explainer_name in EXPLAINERS:
        print(f"\n{'='*70}")
        print(f"Benchmark: {explainer_name}")
        print(f"{'='*70}\n")
        
        try:
            result = run_distributed_benchmark(
                model_path=MODEL_PATH,
                tasks=tasks,
                explainer_name=explainer_name,
                num_workers=NUM_WORKERS,
                explainer_config=EXPLAINER_CONFIGS[explainer_name],
                device=DEVICE
            )
            
            all_results[explainer_name] = result
            
            # Print progress summary
            print(f"\n{'='*70}")
            print(f"Progress Summary")
            print(f"{'='*70}")
            for exp_name, exp_result in all_results.items():
                makespan = exp_result['execution_time']
                success = exp_result['successful_tasks']
                timeout = exp_result.get('timeout_tasks', 0)
                total = success + timeout + exp_result['failed_tasks']
                print(f"{exp_name:15s} | Makespan: {makespan:7.2f}s | Success: {success:3d}/{total:3d} | Timeout: {timeout:3d}")
            
        except Exception as e:
            print(f"\nERROR in benchmark ({explainer_name}): {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with next explainer...\n")
    
    # Save results
    output_file = 'results/treecycle_distributed_benchmark.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        # Remove detailed results for JSON
        summary = {k: {kk: vv for kk, vv in v.items() if kk != 'results'} 
                  for k, v in all_results.items()}
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*70)
    print("All benchmarks completed!")
    print("="*70)


if __name__ == '__main__':
    main()
