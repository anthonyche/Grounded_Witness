"""
Distributed benchmark for TreeCycle graph: sample 100 target nodes, induce L-hop subgraphs, distribute to 20 workers, run explainers (ApxChase, HeuChase, GNNExplainer).
"""
import torch
import random
import os
import multiprocessing as mp
from torch_geometric.utils import k_hop_subgraph

# 配置参数
GRAPH_PATH = 'datasets/TreeCycle/treecycle_d5_bf15_n813616.pt'
MODEL_PATH = 'models/TreeCycle_gcn_d5_bf15_n813616.pth'
NUM_TARGETS = 100
L_HOP = 2
NUM_WORKERS = 20
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# 1. 加载图和模型
data = torch.load(GRAPH_PATH)
all_nodes = list(range(data.num_nodes))
model = None  # 这里可以加载GCN模型，如果解释器需要

# 2. 随机采样 target nodes
target_nodes = random.sample(all_nodes, NUM_TARGETS)

# 3. Induce L-hop subgraphs
def induce_subgraph(node_id):
    subset, edge_index, _, _ = k_hop_subgraph(
        node_id, L_HOP, data.edge_index, relabel_nodes=True)
    sub_data = data.__class__()
    sub_data.x = data.x[subset]
    sub_data.y = data.y[subset]
    sub_data.edge_index = edge_index
    sub_data.num_nodes = len(subset)
    sub_data.target = node_id
    return sub_data

subgraphs = [induce_subgraph(n) for n in target_nodes]

# 4. 分发到 20 个 worker
chunks = [[] for _ in range(NUM_WORKERS)]
for i, sg in enumerate(subgraphs):
    chunks[i % NUM_WORKERS].append(sg)

# 5. Worker 解释函数（可扩展）
def run_explainers(subgraph_list, worker_id):
    results = []
    for sub_data in subgraph_list:
        # 这里可以插入 ApxChase, HeuChase, GNNExplainer 等解释器
        # 这里只做伪代码占位
        result = {
            'target': sub_data.target,
            'num_nodes': sub_data.num_nodes,
            'apxchase': 'done',
            'heuchase': 'done',
            'gnnexplainer': 'done'
        }
        results.append(result)
    print(f'Worker {worker_id} finished {len(subgraph_list)} subgraphs.')
    return results

# 6. 多进程并行
if __name__ == '__main__':
    with mp.Pool(NUM_WORKERS) as pool:
        worker_args = [(chunks[i], i) for i in range(NUM_WORKERS)]
        all_results = pool.starmap(run_explainers, worker_args)
    # 汇总结果
    flat_results = [r for worker in all_results for r in worker]
    print(f'Total targets explained: {len(flat_results)}')
    # 保存结果
    torch.save(flat_results, 'results/treecycle_benchmark_results.pt')
    print('Results saved to results/treecycle_benchmark_results.pt')
