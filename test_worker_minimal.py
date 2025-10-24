#!/usr/bin/env python3
"""
最小化测试：验证 multiprocessing worker 能否启动
"""

import multiprocessing as mp
import sys
import os
import time

def minimal_worker(worker_id, result_queue):
    """最小化 worker：只打印和返回"""
    # 强制刷新输出
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    print(f"\n{'='*40}", flush=True)
    print(f"WORKER {worker_id}: STARTED (PID: {os.getpid()})", flush=True)
    print(f"{'='*40}\n", flush=True)
    
    # 模拟一些工作
    time.sleep(0.5)
    
    # 发送结果
    result_queue.put({
        'worker_id': worker_id,
        'status': 'success',
        'message': f'Worker {worker_id} completed'
    })
    
    print(f"Worker {worker_id}: Sent result to queue", flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("="*70)
    print("MINIMAL WORKER TEST")
    print("="*70)
    
    num_workers = 5
    result_queue = mp.Queue()
    
    # 启动 workers
    processes = []
    print(f"\nLaunching {num_workers} workers...")
    for i in range(num_workers):
        print(f"  Starting worker {i}...")
        p = mp.Process(target=minimal_worker, args=(i, result_queue))
        p.start()
        processes.append(p)
        print(f"  Worker {i} launched (PID: {p.pid})")
        time.sleep(0.1)
    
    print(f"\nWaiting for {num_workers} results...")
    
    # 收集结果
    results = []
    for i in range(num_workers):
        print(f"  Waiting for result {i+1}/{num_workers}...")
        result = result_queue.get(timeout=10)
        results.append(result)
        print(f"  ✓ Received: {result['message']}")
    
    # 等待进程完成
    for p in processes:
        p.join()
    
    print(f"\n✓ All {num_workers} workers completed successfully!")
    print("="*70)
