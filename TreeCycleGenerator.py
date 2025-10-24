"""
Tree-Cycle Graph Generator

生成 Tree-Cycle 结构的合成图，用于 scalability 测试。

Tree-Cycle 结构：
- 树结构：从根节点开始，每个节点有 branching_factor 个子节点
- 环结构：在每一层的节点之间添加环边，形成循环
- 混合结构：树的层次性 + 环的循环性，适合测试约束传播

参数：
- depth: 树的深度
- branching_factor: 分支因子（每个节点的子节点数）
- cycle_prob: 在同层节点间添加环边的概率
- num_node_types: 节点类型数量（用于约束定义）

数据集大小：
- 小规模（本地测试）：depth=3, branching_factor=5, ~156 nodes
- 中等规模：depth=4, branching_factor=10, ~11,111 nodes
- 大规模（HPC）：depth=6, branching_factor=20, ~8.4M nodes
- 超大规模：depth=7, branching_factor=30, ~1B nodes
"""

import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from collections import defaultdict


class TreeCycleGenerator:
    """Tree-Cycle 图生成器"""
    
    def __init__(self, depth, branching_factor, cycle_prob=0.3, num_node_types=5, seed=42):
        """
        Args:
            depth: 树的深度（层数）
            branching_factor: 每个节点的子节点数
            cycle_prob: 同层节点间添加环边的概率
            num_node_types: 节点类型数量
            seed: 随机种子
        """
        self.depth = depth
        self.branching_factor = branching_factor
        self.cycle_prob = cycle_prob
        self.num_node_types = num_node_types
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 计算节点数量（几何级数求和）
        if branching_factor == 1:
            self.num_nodes = depth + 1
        else:
            self.num_nodes = (branching_factor ** (depth + 1) - 1) // (branching_factor - 1)
        
        print(f"TreeCycle Configuration:")
        print(f"  Depth: {depth}")
        print(f"  Branching factor: {branching_factor}")
        print(f"  Cycle probability: {cycle_prob}")
        print(f"  Node types: {num_node_types}")
        print(f"  Expected nodes: {self.num_nodes:,}")
        
        # 内存估算
        estimated_memory_gb = self.num_nodes * 50 / 1e9  # 粗略估算
        print(f"  Estimated memory: {estimated_memory_gb:.1f} GB")
    
    def generate(self):
        """生成 Tree-Cycle 图"""
        print("\nGenerating Tree-Cycle graph...")
        
        # 1. 构建树结构
        edge_list = []
        node_labels = []  # 节点类型标签
        level_nodes = defaultdict(list)  # 每层的节点列表
        
        node_id = 0
        current_level_nodes = [node_id]  # 根节点
        level_nodes[0].append(node_id)
        node_labels.append(np.random.randint(0, self.num_node_types))
        node_id += 1
        
        # 逐层构建树
        for level in range(1, self.depth + 1):
            next_level_nodes = []
            for parent in current_level_nodes:
                # 为每个父节点添加子节点
                for _ in range(self.branching_factor):
                    child = node_id
                    edge_list.append([parent, child])  # 父→子（树边）
                    
                    # 随机分配节点类型（可以根据层数设计规则）
                    # 这里简单地随机分配，也可以设计特定模式
                    node_type = np.random.randint(0, self.num_node_types)
                    node_labels.append(node_type)
                    
                    level_nodes[level].append(child)
                    next_level_nodes.append(child)
                    node_id += 1
            
            current_level_nodes = next_level_nodes
            print(f"  Level {level}: {len(current_level_nodes):,} nodes")
        
        print(f"  Total tree edges: {len(edge_list):,}")
        
        # 2. 添加环边（同层节点间的循环边）
        cycle_edges = 0
        for level, nodes in level_nodes.items():
            if len(nodes) <= 1:
                continue
            
            # 在同层节点间随机添加环边
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if np.random.rand() < self.cycle_prob:
                        # 双向边（无向图）
                        edge_list.append([nodes[i], nodes[j]])
                        edge_list.append([nodes[j], nodes[i]])
                        cycle_edges += 2
        
        print(f"  Cycle edges added: {cycle_edges:,}")
        print(f"  Total edges: {len(edge_list):,}")
        
        # 3. 转换为 PyG Data 格式
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_labels, dtype=torch.long).unsqueeze(1)  # Node features (label as feature)
        
        # 创建 one-hot 编码的节点特征
        x_onehot = torch.zeros(self.num_nodes, self.num_node_types)
        x_onehot[torch.arange(self.num_nodes), x.squeeze()] = 1
        
        data = Data(
            x=x_onehot,
            edge_index=edge_index,
            y=x.squeeze(),  # 使用类型作为标签
            num_nodes=self.num_nodes,
        )
        
        # 添加元数据
        data.depth = self.depth
        data.branching_factor = self.branching_factor
        data.cycle_prob = self.cycle_prob
        data.num_node_types = self.num_node_types
        data.level_nodes = dict(level_nodes)  # 每层的节点
        
        print(f"\n✓ Tree-Cycle graph generated:")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.edge_index.shape[1]:,}")
        print(f"  Node types: {self.num_node_types}")
        print(f"  Features: {data.x.shape}")
        
        return data
    
    def visualize(self, data, output_file='treecycle_graph.png', max_nodes=200):
        """可视化 Tree-Cycle 图（仅适用于小图）"""
        if data.num_nodes > max_nodes:
            print(f"Graph too large ({data.num_nodes} nodes), skipping visualization (max: {max_nodes})")
            return
        
        print(f"\nVisualizing Tree-Cycle graph...")
        
        # 转换为 NetworkX 图
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # 使用层次布局
        pos = {}
        level_nodes = data.level_nodes
        
        for level, nodes in level_nodes.items():
            y = -level  # Y 坐标按层递减
            num_nodes_in_level = len(nodes)
            x_spacing = 1.0 / max(num_nodes_in_level, 1)
            
            for i, node in enumerate(nodes):
                x = (i - num_nodes_in_level / 2) * x_spacing
                pos[node] = (x, y)
        
        # 节点颜色（按类型）
        node_colors = data.y.numpy()
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              cmap=plt.cm.Set3, node_size=100, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        
        ax.set_title(f'Tree-Cycle Graph (depth={self.depth}, bf={self.branching_factor}, nodes={data.num_nodes})')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_file}")
        plt.close()
    
    def save(self, data, output_dir='datasets/TreeCycle'):
        """保存生成的图数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"treecycle_d{self.depth}_bf{self.branching_factor}_n{data.num_nodes}.pt"
        filepath = os.path.join(output_dir, filename)
        
        torch.save(data, filepath)
        print(f"\n✓ Saved graph data: {filepath}")
        
        # 保存统计信息
        stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'depth': self.depth,
            'branching_factor': self.branching_factor,
            'cycle_prob': self.cycle_prob,
            'num_node_types': self.num_node_types,
        }
        
        stats_file = filepath.replace('.pt', '_stats.pkl')
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        return filepath


def load_treecycle(filepath):
    """加载 Tree-Cycle 图数据"""
    print(f"Loading Tree-Cycle graph from: {filepath}")
    data = torch.load(filepath)
    
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    print(f"  Depth: {data.depth}")
    print(f"  Branching factor: {data.branching_factor}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Generate Tree-Cycle synthetic graphs')
    parser.add_argument('--depth', type=int, default=3, 
                       help='Tree depth (default: 3, ~156 nodes for bf=5)')
    parser.add_argument('--branching-factor', type=int, default=5,
                       help='Branching factor (default: 5)')
    parser.add_argument('--cycle-prob', type=float, default=0.3,
                       help='Probability of adding cycle edges (default: 0.3)')
    parser.add_argument('--num-types', type=int, default=5,
                       help='Number of node types (default: 5)')
    parser.add_argument('--output-dir', type=str, default='datasets/TreeCycle',
                       help='Output directory (default: datasets/TreeCycle)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the graph (only for small graphs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Tree-Cycle Graph Generator")
    print("="*70)
    
    # 生成图
    generator = TreeCycleGenerator(
        depth=args.depth,
        branching_factor=args.branching_factor,
        cycle_prob=args.cycle_prob,
        num_node_types=args.num_types,
        seed=args.seed
    )
    
    data = generator.generate()
    
    # 可视化（仅小图）
    if args.visualize and data.num_nodes <= 200:
        generator.visualize(data)
    
    # 保存
    filepath = generator.save(data, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)
    print(f"\nTo load this graph:")
    print(f"  from TreeCycleGenerator import load_treecycle")
    print(f"  data = load_treecycle('{filepath}')")
    
    # 打印不同规模的配置建议
    print("\n" + "="*70)
    print("Suggested configurations for different scales:")
    print("="*70)
    print("Small (local testing):")
    print("  python TreeCycleGenerator.py --depth 3 --branching-factor 5")
    print("  → ~156 nodes")
    print("\nMedium:")
    print("  python TreeCycleGenerator.py --depth 4 --branching-factor 10")
    print("  → ~11,111 nodes")
    print("\nLarge (HPC):")
    print("  python TreeCycleGenerator.py --depth 6 --branching-factor 20")
    print("  → ~8.4M nodes")
    print("\nBillion-scale (HPC):")
    print("  python TreeCycleGenerator.py --depth 7 --branching-factor 30")
    print("  → ~1B nodes")


if __name__ == '__main__':
    main()
