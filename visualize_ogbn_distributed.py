"""
Visualize distributed benchmark results for OGBN-Papers100M
"""

import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_results(results_dir='results/ogbn_distributed'):
    """加载所有实验结果"""
    results_file = os.path.join(results_dir, 'complete_results.pkl')
    
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Load from individual JSON files
        results = []
        for json_file in Path(results_dir).glob('*.json'):
            if json_file.name != 'complete_results.pkl':
                with open(json_file, 'r') as f:
                    results.append(json.load(f))
        return results


def plot_runtime_vs_workers(results, save_path='results/ogbn_distributed'):
    """绘制运行时间 vs 处理器数量"""
    
    explainers = sorted(set(r['explainer'] for r in results))
    num_workers_list = sorted(set(r['num_workers'] for r in results))
    
    plt.figure(figsize=(12, 6))
    
    # Plot total time
    plt.subplot(1, 2, 1)
    for explainer in explainers:
        exp_results = [r for r in results if r['explainer'] == explainer]
        exp_results = sorted(exp_results, key=lambda x: x['num_workers'])
        
        workers = [r['num_workers'] for r in exp_results]
        times = [r['total_time'] for r in exp_results]
        
        plt.plot(workers, times, marker='o', label=explainer, linewidth=2, markersize=8)
    
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Total Runtime (seconds)', fontsize=12)
    plt.title('Runtime vs Number of Workers', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    for explainer in explainers:
        exp_results = [r for r in results if r['explainer'] == explainer]
        exp_results = sorted(exp_results, key=lambda x: x['num_workers'])
        
        workers = [r['num_workers'] for r in exp_results]
        speedups = [r['worker_time_mean'] / r['parallel_time'] for r in exp_results]
        
        plt.plot(workers, speedups, marker='s', label=explainer, linewidth=2, markersize=8)
    
    # Plot ideal speedup
    plt.plot(num_workers_list, num_workers_list, 'k--', label='Ideal', linewidth=1.5)
    
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Number of Workers', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'runtime_vs_workers.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_path, 'runtime_vs_workers.png')}")
    plt.close()


def plot_load_balance(results, save_path='results/ogbn_distributed'):
    """绘制负载均衡效果"""
    
    explainers = sorted(set(r['explainer'] for r in results))
    
    plt.figure(figsize=(14, 5))
    
    for idx, explainer in enumerate(explainers):
        exp_results = [r for r in results if r['explainer'] == explainer]
        exp_results = sorted(exp_results, key=lambda x: x['num_workers'])
        
        plt.subplot(1, len(explainers), idx + 1)
        
        workers_list = [r['num_workers'] for r in exp_results]
        balance_ratios = [r['load_balance_ratio'] for r in exp_results]
        
        plt.bar(range(len(workers_list)), balance_ratios, color='steelblue', alpha=0.7)
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Balance')
        
        plt.xticks(range(len(workers_list)), workers_list)
        plt.xlabel('Number of Workers', fontsize=11)
        plt.ylabel('Load Balance Ratio', fontsize=11)
        plt.title(f'{explainer}', fontsize=12, fontweight='bold')
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'load_balance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_path, 'load_balance.png')}")
    plt.close()


def plot_efficiency(results, save_path='results/ogbn_distributed'):
    """绘制并行效率"""
    
    explainers = sorted(set(r['explainer'] for r in results))
    
    plt.figure(figsize=(10, 6))
    
    for explainer in explainers:
        exp_results = [r for r in results if r['explainer'] == explainer]
        exp_results = sorted(exp_results, key=lambda x: x['num_workers'])
        
        workers = [r['num_workers'] for r in exp_results]
        speedups = [r['worker_time_mean'] / r['parallel_time'] for r in exp_results]
        efficiencies = [speedup / w * 100 for speedup, w in zip(speedups, workers)]
        
        plt.plot(workers, efficiencies, marker='D', label=explainer, linewidth=2, markersize=8)
    
    plt.axhline(y=100, color='black', linestyle='--', linewidth=1.5, label='100% Efficiency')
    
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Parallel Efficiency (%)', fontsize=12)
    plt.title('Parallel Efficiency vs Number of Workers', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'parallel_efficiency.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_path, 'parallel_efficiency.png')}")
    plt.close()


def generate_summary_table(results, save_path='results/ogbn_distributed'):
    """生成结果摘要表"""
    
    explainers = sorted(set(r['explainer'] for r in results))
    num_workers_list = sorted(set(r['num_workers'] for r in results))
    
    # Create markdown table
    table_lines = []
    table_lines.append("# Distributed Benchmark Results Summary\n")
    table_lines.append(f"**Dataset**: ogbn-papers100M")
    table_lines.append(f"**Number of test nodes**: {results[0]['num_nodes']}")
    table_lines.append(f"**Number of hops**: {results[0]['num_hops']}\n")
    
    # Runtime table
    table_lines.append("## Runtime (seconds)\n")
    table_lines.append("| Explainer | " + " | ".join([f"{w} workers" for w in num_workers_list]) + " |")
    table_lines.append("|" + "---|" * (len(num_workers_list) + 1))
    
    for explainer in explainers:
        row = [explainer]
        for num_workers in num_workers_list:
            r = next((r for r in results if r['explainer'] == explainer and r['num_workers'] == num_workers), None)
            if r:
                row.append(f"{r['total_time']:.2f}")
            else:
                row.append("N/A")
        table_lines.append("| " + " | ".join(row) + " |")
    
    # Speedup table
    table_lines.append("\n## Speedup\n")
    table_lines.append("| Explainer | " + " | ".join([f"{w} workers" for w in num_workers_list]) + " |")
    table_lines.append("|" + "---|" * (len(num_workers_list) + 1))
    
    for explainer in explainers:
        row = [explainer]
        for num_workers in num_workers_list:
            r = next((r for r in results if r['explainer'] == explainer and r['num_workers'] == num_workers), None)
            if r:
                speedup = r['worker_time_mean'] / r['parallel_time']
                row.append(f"{speedup:.2f}x")
            else:
                row.append("N/A")
        table_lines.append("| " + " | ".join(row) + " |")
    
    # Efficiency table
    table_lines.append("\n## Parallel Efficiency (%)\n")
    table_lines.append("| Explainer | " + " | ".join([f"{w} workers" for w in num_workers_list]) + " |")
    table_lines.append("|" + "---|" * (len(num_workers_list) + 1))
    
    for explainer in explainers:
        row = [explainer]
        for num_workers in num_workers_list:
            r = next((r for r in results if r['explainer'] == explainer and r['num_workers'] == num_workers), None)
            if r:
                speedup = r['worker_time_mean'] / r['parallel_time']
                efficiency = speedup / num_workers * 100
                row.append(f"{efficiency:.1f}%")
            else:
                row.append("N/A")
        table_lines.append("| " + " | ".join(row) + " |")
    
    # Load balance table
    table_lines.append("\n## Load Balance Ratio\n")
    table_lines.append("| Explainer | " + " | ".join([f"{w} workers" for w in num_workers_list]) + " |")
    table_lines.append("|" + "---|" * (len(num_workers_list) + 1))
    
    for explainer in explainers:
        row = [explainer]
        for num_workers in num_workers_list:
            r = next((r for r in results if r['explainer'] == explainer and r['num_workers'] == num_workers), None)
            if r:
                row.append(f"{r['load_balance_ratio']:.3f}")
            else:
                row.append("N/A")
        table_lines.append("| " + " | ".join(row) + " |")
    
    # Save to file
    summary_file = os.path.join(save_path, 'RESULTS_SUMMARY.md')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"Saved: {summary_file}")
    
    # Also print to console
    print("\n" + '\n'.join(table_lines))


def main():
    """主函数"""
    
    results_dir = 'results/ogbn_distributed'
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} benchmark results")
    
    print("\nGenerating visualizations...")
    plot_runtime_vs_workers(results, results_dir)
    plot_load_balance(results, results_dir)
    plot_efficiency(results, results_dir)
    
    print("\nGenerating summary table...")
    generate_summary_table(results, results_dir)
    
    print("\nAll visualizations completed!")


if __name__ == '__main__':
    main()
