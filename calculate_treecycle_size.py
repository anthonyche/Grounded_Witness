#!/usr/bin/env python3
"""
TreeCycle 图大小快速计算脚本

用法：
  python calculate_treecycle_size.py --depth 5 --bf 15 --cycle-prob 0.2
  python calculate_treecycle_size.py --list-configs
"""

import argparse


def calculate_treecycle_size(depth, branching_factor, cycle_prob):
    """计算 TreeCycle 图的节点数和边数"""
    b = branching_factor
    d = depth
    
    # 节点数
    if b == 1:
        num_nodes = d + 1
    else:
        num_nodes = (b ** (d + 1) - 1) // (b - 1)
    
    # 树边
    tree_edges = num_nodes - 1
    
    # 环边估算
    cycle_edges = 0
    for level in range(d + 1):
        nodes_in_level = b ** level
        if nodes_in_level > 1:
            max_pairs = nodes_in_level * (nodes_in_level - 1) // 2
            actual_pairs = int(max_pairs * cycle_prob)
            actual_pairs = min(actual_pairs, nodes_in_level * 10)
            cycle_edges += actual_pairs * 2
    
    total_edges = tree_edges + cycle_edges
    
    # 内存估算（GB）
    memory_gb = (2 * total_edges * 8 + num_nodes * 5 * 4) / 1e9
    
    return {
        'num_nodes': num_nodes,
        'tree_edges': tree_edges,
        'cycle_edges': cycle_edges,
        'total_edges': total_edges,
        'memory_gb': memory_gb,
    }


def format_number(n):
    """格式化大数字"""
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n/1000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M"
    else:
        return f"{n/1_000_000_000:.1f}B"


def print_config(name, depth, bf, cycle_prob, description=""):
    """打印配置信息"""
    stats = calculate_treecycle_size(depth, bf, cycle_prob)
    
    print(f"\n{'='*70}")
    print(f"{name}: {description}")
    print(f"{'='*70}")
    print(f"配置: depth={depth}, branching_factor={bf}, cycle_prob={cycle_prob}")
    print(f"节点数: {stats['num_nodes']:,} ({format_number(stats['num_nodes'])})")
    print(f"边数: {stats['total_edges']:,} ({format_number(stats['total_edges'])})")
    print(f"  - 树边: {stats['tree_edges']:,}")
    print(f"  - 环边: {stats['cycle_edges']:,}")
    print(f"估算内存: {stats['memory_gb']:.2f} GB")
    
    # 时间和资源建议
    if stats['num_nodes'] < 100_000:
        print(f"建议时间: ~10-30 分钟")
        print(f"建议内存: 16-32 GB")
    elif stats['num_nodes'] < 1_000_000:
        print(f"建议时间: ~30-60 分钟")
        print(f"建议内存: 64 GB")
    elif stats['num_nodes'] < 20_000_000:
        print(f"建议时间: ~1-2 小时")
        print(f"建议内存: 128 GB")
    elif stats['num_nodes'] < 100_000_000:
        print(f"建议时间: ~2-4 小时")
        print(f"建议内存: 256 GB")
    else:
        print(f"建议时间: ~4-8 小时")
        print(f"建议内存: 512 GB")
    
    # 生成命令
    print(f"\n生成命令:")
    print(f"  python TreeCycleGenerator.py --depth {depth} --branching-factor {bf} --cycle-prob {cycle_prob}")


def list_all_configs():
    """列出所有推荐配置"""
    print("="*70)
    print("TreeCycle 图推荐配置")
    print("="*70)
    
    configs = [
        {
            'name': '小规模-1',
            'depth': 4,
            'bf': 15,
            'cycle_prob': 0.2,
            'description': '快速测试（约54K节点）'
        },
        {
            'name': '小规模-2',
            'depth': 5,
            'bf': 10,
            'cycle_prob': 0.2,
            'description': '中等测试（约111K节点）'
        },
        {
            'name': '当前配置',
            'depth': 5,
            'bf': 15,
            'cycle_prob': 0.2,
            'description': '标准实验（约813K节点）✅'
        },
        {
            'name': '大规模-1',
            'depth': 6,
            'bf': 15,
            'cycle_prob': 0.15,
            'description': 'Scalability测试（约12M节点）'
        },
        {
            'name': '大规模-2',
            'depth': 6,
            'bf': 20,
            'cycle_prob': 0.15,
            'description': '超大规模测试（约67M节点）'
        },
        {
            'name': '超大规模',
            'depth': 6,
            'bf': 25,
            'cycle_prob': 0.1,
            'description': '极限测试（约254M节点）'
        },
    ]
    
    for config in configs:
        print_config(
            config['name'],
            config['depth'],
            config['bf'],
            config['cycle_prob'],
            config['description']
        )
    
    # 打印对比表
    print("\n" + "="*70)
    print("配置对比表")
    print("="*70)
    print(f"{'配置':<12} {'Depth':<6} {'BF':<4} {'CP':<5} {'节点数':<12} {'边数':<12} {'内存(GB)':<10}")
    print("-"*70)
    
    for config in configs:
        stats = calculate_treecycle_size(config['depth'], config['bf'], config['cycle_prob'])
        print(f"{config['name']:<12} "
              f"{config['depth']:<6} "
              f"{config['bf']:<4} "
              f"{config['cycle_prob']:<5.2f} "
              f"{format_number(stats['num_nodes']):<12} "
              f"{format_number(stats['total_edges']):<12} "
              f"{stats['memory_gb']:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description='计算 TreeCycle 图大小')
    parser.add_argument('--depth', type=int, help='树的深度')
    parser.add_argument('--bf', '--branching-factor', type=int, dest='bf', help='分支因子')
    parser.add_argument('--cycle-prob', type=float, help='环边概率')
    parser.add_argument('--list-configs', action='store_true', help='列出所有推荐配置')
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_all_configs()
    elif args.depth and args.bf and args.cycle_prob:
        print_config(
            "自定义配置",
            args.depth,
            args.bf,
            args.cycle_prob
        )
    else:
        print("错误: 请提供 --depth, --bf, --cycle-prob 或使用 --list-configs")
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
