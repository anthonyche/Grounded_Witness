#!/usr/bin/env python3
"""
收集并对比所有baseline方法的Fidelity-统计数据
从已保存的metrics_graph_*.json文件中读取数据
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List

def collect_fidelity_stats(results_dir: str, method_name: str) -> Dict:
    """
    从指定目录收集所有图的fidelity-, conciseness和coverage统计
    
    Args:
        results_dir: 结果目录路径
        method_name: 方法名称
    
    Returns:
        包含统计信息的字典
    """
    metrics_files = glob.glob(os.path.join(results_dir, "metrics_graph_*.json"))
    
    if not metrics_files:
        print(f"警告: {results_dir} 中没有找到metrics文件")
        return None
    
    all_fidelities = []
    graph_fidelities = []
    all_conciseness = []
    graph_conciseness = []
    graph_coverage = []
    
    for metrics_file in sorted(metrics_files):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 获取这个图的平均fidelity
        if 'avg_fidelity_minus' in data:
            graph_avg_fid = data['avg_fidelity_minus']
            graph_fidelities.append(graph_avg_fid)
            
            # 也收集每个witness的fidelity
            if 'witnesses' in data:
                for witness in data['witnesses']:
                    if 'fidelity_minus' in witness:
                        all_fidelities.append(witness['fidelity_minus'])
        
        # 获取conciseness
        if 'avg_conciseness' in data:
            graph_avg_conc = data['avg_conciseness']
            graph_conciseness.append(graph_avg_conc)
            
            # 也收集每个witness的conciseness
            if 'witnesses' in data:
                for witness in data['witnesses']:
                    if 'conciseness' in witness:
                        all_conciseness.append(witness['conciseness'])
        
        # 获取coverage
        if 'coverage_ratio' in data:
            graph_coverage.append(data['coverage_ratio'])
    
    if not graph_fidelities:
        print(f"警告: {results_dir} 中的metrics文件不包含fidelity数据")
        return None
    
    result = {
        'method': method_name,
        'num_graphs': len(graph_fidelities),
        'num_explanations': len(all_fidelities),
        'overall_avg_fidelity': float(np.mean(graph_fidelities)),
        'explanation_avg_fidelity': float(np.mean(all_fidelities)) if all_fidelities else 0.0,
        'min_fidelity': float(min(graph_fidelities)),
        'max_fidelity': float(max(graph_fidelities)),
        'std_fidelity': float(np.std(graph_fidelities)),
        'per_graph_fidelities': graph_fidelities,
        'overall_avg_conciseness': float(np.mean(graph_conciseness)) if graph_conciseness else 0.0,
        'explanation_avg_conciseness': float(np.mean(all_conciseness)) if all_conciseness else 0.0,
        'min_conciseness': float(min(graph_conciseness)) if graph_conciseness else 0.0,
        'max_conciseness': float(max(graph_conciseness)) if graph_conciseness else 0.0,
        'std_conciseness': float(np.std(graph_conciseness)) if graph_conciseness else 0.0,
        'per_graph_conciseness': graph_conciseness,
    }
    
    # 添加coverage统计（如果有的话）
    if graph_coverage:
        result.update({
            'overall_avg_coverage': float(np.mean(graph_coverage)),
            'min_coverage': float(min(graph_coverage)),
            'max_coverage': float(max(graph_coverage)),
            'std_coverage': float(np.std(graph_coverage)),
            'per_graph_coverage': graph_coverage,
        })
    
    return result

def main():
    base_dir = "/Users/anthonyche/Desktop/Research/GroundingGEXP/src/results/MUTAG"
    
    methods = [
        ("apxchase_mutag", "ApxChase"),
        ("exhaustchase_mutag", "ExhaustChase"),
        ("heuchase_mutag", "HeuChase"),
        ("gnnexplainer_mutag", "GNNExplainer"),
        ("pgexplainer_mutag", "PGExplainer"),
    ]
    
    print("=" * 80)
    print("解释质量统计对比 (Fidelity- & Conciseness)")
    print("=" * 80)
    print()
    
    all_stats = []
    
    for dir_name, method_name in methods:
        results_dir = os.path.join(base_dir, dir_name)
        
        if not os.path.exists(results_dir):
            print(f"⚠️  {method_name}: 目录不存在 ({results_dir})")
            print()
            continue
        
        stats = collect_fidelity_stats(results_dir, method_name)
        
        if stats:
            all_stats.append(stats)
            print(f"✅ {method_name}:")
            print(f"   图数量: {stats['num_graphs']}")
            print(f"   解释数量: {stats['num_explanations']}")
            print(f"   Overall Avg Fidelity-: {stats['overall_avg_fidelity']:.6f}")
            print(f"   Explanation Avg Fidelity-: {stats['explanation_avg_fidelity']:.6f}")
            print(f"   Min/Max/Std Fidelity-: {stats['min_fidelity']:.6f} / {stats['max_fidelity']:.6f} / {stats['std_fidelity']:.6f}")
            if stats['overall_avg_conciseness'] > 0:
                print(f"   Overall Avg Conciseness: {stats['overall_avg_conciseness']:.6f}")
                print(f"   Explanation Avg Conciseness: {stats['explanation_avg_conciseness']:.6f}")
                print(f"   Min/Max/Std Conciseness: {stats['min_conciseness']:.6f} / {stats['max_conciseness']:.6f} / {stats['std_conciseness']:.6f}")
            if 'overall_avg_coverage' in stats:
                print(f"   Overall Avg Coverage: {stats['overall_avg_coverage']:.6f} ({stats['overall_avg_coverage']*100:.2f}%)")
                print(f"   Min/Max/Std Coverage: {stats['min_coverage']:.6f} / {stats['max_coverage']:.6f} / {stats['std_coverage']:.6f}")
            print()
    
    if len(all_stats) > 1:
        print("=" * 80)
        print("排名对比 (按Overall Avg Fidelity-降序)")
        print("=" * 80)
        
        # 按fidelity降序排序（更高的fidelity-意味着解释更重要）
        sorted_stats = sorted(all_stats, key=lambda x: x['overall_avg_fidelity'], reverse=True)
        
        for i, stats in enumerate(sorted_stats, 1):
            fid = stats['overall_avg_fidelity']
            conc = stats.get('overall_avg_conciseness', 0.0)
            print(f"{i}. {stats['method']}: Fidelity-={fid:.6f}, Conciseness={conc:.6f}")
        print()
        
        # 按Conciseness降序排序（更高的conciseness表示更简洁）
        if any(s.get('overall_avg_conciseness', 0) > 0 for s in all_stats):
            print("=" * 80)
            print("排名对比 (按Overall Avg Conciseness降序)")
            print("=" * 80)
            
            sorted_by_conc = sorted(all_stats, key=lambda x: x.get('overall_avg_conciseness', 0.0), reverse=True)
            
            for i, stats in enumerate(sorted_by_conc, 1):
                fid = stats['overall_avg_fidelity']
                conc = stats.get('overall_avg_conciseness', 0.0)
                print(f"{i}. {stats['method']}: Conciseness={conc:.6f}, Fidelity-={fid:.6f}")
            print()
        
        # 按Coverage降序排序（如果有coverage数据）
        if any('overall_avg_coverage' in s for s in all_stats):
            print("=" * 80)
            print("排名对比 (按Coverage降序)")
            print("=" * 80)
            
            sorted_by_cov = sorted(
                [s for s in all_stats if 'overall_avg_coverage' in s], 
                key=lambda x: x['overall_avg_coverage'], 
                reverse=True
            )
            
            for i, stats in enumerate(sorted_by_cov, 1):
                cov = stats['overall_avg_coverage']
                conc = stats.get('overall_avg_conciseness', 0.0)
                fid = stats['overall_avg_fidelity']
                print(f"{i}. {stats['method']}: Coverage={cov*100:.2f}%, Conciseness={conc:.4f}, Fidelity-={fid:.4f}")
            print()
        
        print("=" * 80)
        print("指标说明:")
        print("=" * 80)
        print("• Fidelity- = Pr(M(G)) - Pr(M(G_s))")
        print("  - 正值: 解释子图的预测概率低于原图 (移除了重要结构)")
        print("  - 负值: 解释子图的预测概率高于原图 (移除了噪声或干扰)")
        print("  - 绝对值越大表示解释的影响越显著")
        print()
        print("• Conciseness = 1 - (解释边数 / 原图边数)")
        print("  - 值越高表示解释越简洁 (使用更少的边)")
        print("  - 范围: [0, 1]，1表示最简洁")
        print()
        print("• Coverage = |覆盖的约束| / |总约束数|")
        print("  - 值越高表示解释覆盖了更多的约束条件")
        print("  - 范围: [0, 1]，1表示覆盖所有约束")
        print("  - 使用相同的matching策略 (head matches + repair cost <= Budget)")
        print()

if __name__ == "__main__":
    main()
