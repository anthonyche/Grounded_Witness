# TreeCycle Distributed Benchmark - Signal/Timeout Bug Fix

## Problem
Workers were hanging at `heuchase._run()` call in multiprocessing environment, even though HeuChase runs fast when tested standalone.

## Root Cause
**`signal.alarm()` does not work correctly with Python multiprocessing!**

The TreeCycle version had:
```python
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Task timeout")

# In worker_process:
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_seconds)  # Set 30min timeout
```

This causes deadlocks in multiprocessing because:
1. Signal handlers are process-global
2. `spawn` method creates new processes that don't inherit signal handlers correctly
3. SIGALRM can interfere with multiprocessing's internal machinery
4. Workers get stuck waiting for signals that never arrive

## Solution
**Remove ALL signal/alarm code**, matching OGBN's approach:

```python
# REMOVED:
# import signal
# class TimeoutException
# def timeout_handler
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(timeout_seconds)
# signal.alarm(0)
# except TimeoutException

# KEPT: Simple try-except like OGBN
try:
    Sigma_star, S_k = explainer._run(H=subgraph, root=int(target_node))
    # ... process results
except Exception as e:
    print(f"Worker {worker_id}: Error: {e}")
    explanation_result = {'success': False, 'error': str(e)}
```

## Key Changes
1. **Removed**: `import signal`
2. **Removed**: `TimeoutException` class and `timeout_handler` function
3. **Removed**: `signal.signal()` and `signal.alarm()` calls
4. **Simplified**: Exception handling to match OGBN (no timeout logic)

## Result
- No more hanging at `heuchase._run()`
- Workers can properly communicate via `result_queue`
- Matches OGBN's proven-working architecture

## Files Modified
- `benchmark_treecycle_distributed_v2.py`:
  - Line ~17: Removed `import signal`
  - Lines ~36-41: Removed TimeoutException and timeout_handler
  - Lines ~298-398: Simplified worker task processing (removed all signal.alarm code)

## Testing
Run with small test first:
```bash
python test_treecycle_simplified.py  # 10 nodes, 2 workers
```

Then full benchmark:
```bash
python benchmark_treecycle_distributed_v2.py  # 100 nodes, 20 workers
```
