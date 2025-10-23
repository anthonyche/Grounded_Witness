"""
Cache Management Tool for Distributed Benchmark

Usage:
    python manage_cache.py list          # List all cached subgraphs
    python manage_cache.py info <file>   # Show cache file info
    python manage_cache.py clean         # Remove all cache files
    python manage_cache.py clean-old     # Remove cache files older than 7 days
"""

import os
import sys
import torch
from datetime import datetime, timedelta


def list_cache_files(cache_dir='cache/subgraphs'):
    """列出所有缓存文件"""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
    
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
        return
    
    print(f"Found {len(cache_files)} cache file(s) in {cache_dir}:")
    print(f"\n{'Filename':<60} {'Size':<12} {'Modified':<20}")
    print("-" * 95)
    
    total_size = 0
    for filename in sorted(cache_files):
        filepath = os.path.join(cache_dir, filename)
        size_mb = os.path.getsize(filepath) / 1024**2
        total_size += size_mb
        
        mtime = os.path.getmtime(filepath)
        modified = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{filename:<60} {size_mb:>10.2f} MB {modified}")
    
    print("-" * 95)
    print(f"Total cache size: {total_size:.2f} MB")


def show_cache_info(cache_file):
    """显示缓存文件详细信息"""
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        return
    
    try:
        print(f"Loading cache file: {cache_file}")
        data = torch.load(cache_file)
        
        print("\nCache File Information:")
        print("=" * 60)
        print(f"File size: {os.path.getsize(cache_file) / 1024**2:.2f} MB")
        print(f"Number of tasks: {data['metadata']['num_tasks']}")
        print(f"Number of hops: {data['num_hops']}")
        print(f"Number of node IDs: {len(data['node_ids'])}")
        
        timestamp = data['metadata']['timestamp']
        created = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Created: {created}")
        
        # Task statistics
        tasks = data['tasks']
        edge_counts = [task.num_edges for task in tasks]
        print(f"\nSubgraph Statistics:")
        print(f"  Min edges: {min(edge_counts)}")
        print(f"  Max edges: {max(edge_counts)}")
        print(f"  Mean edges: {sum(edge_counts) / len(edge_counts):.1f}")
        
        # Node IDs sample
        node_ids = data['node_ids']
        print(f"\nNode IDs (first 10): {node_ids[:10]}")
        
    except Exception as e:
        print(f"Error loading cache file: {e}")
        import traceback
        traceback.print_exc()


def clean_cache(cache_dir='cache/subgraphs', days_old=None):
    """清理缓存文件"""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
    
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
        return
    
    now = datetime.now()
    removed_count = 0
    removed_size = 0
    
    for filename in cache_files:
        filepath = os.path.join(cache_dir, filename)
        
        # Check age if days_old is specified
        if days_old is not None:
            mtime = os.path.getmtime(filepath)
            modified = datetime.fromtimestamp(mtime)
            age = now - modified
            
            if age < timedelta(days=days_old):
                continue
        
        # Remove file
        size_mb = os.path.getsize(filepath) / 1024**2
        os.remove(filepath)
        removed_count += 1
        removed_size += size_mb
        print(f"Removed: {filename} ({size_mb:.2f} MB)")
    
    print(f"\nRemoved {removed_count} file(s), freed {removed_size:.2f} MB")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'list':
        list_cache_files()
    
    elif command == 'info':
        if len(sys.argv) < 3:
            print("Usage: python manage_cache.py info <cache_file>")
            sys.exit(1)
        show_cache_info(sys.argv[2])
    
    elif command == 'clean':
        confirm = input("Remove ALL cache files? (yes/no): ")
        if confirm.lower() == 'yes':
            clean_cache()
        else:
            print("Cancelled")
    
    elif command == 'clean-old':
        days = 7
        if len(sys.argv) >= 3:
            days = int(sys.argv[2])
        
        print(f"Removing cache files older than {days} days...")
        clean_cache(days_old=days)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
