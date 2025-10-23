# Distributed Benchmark with Caching

## Overview
This benchmark framework supports **cache-only execution** to avoid loading the full OGBN-Papers100M dataset (111M nodes, 1.6B edges, ~78GB RAM).

## Two-Phase Workflow

### Phase 1: Generate Cache (First Run Only)
```bash
# Option A: Submit cache generation job
sbatch generate_cache.slurm

# Option B: Run directly (if interactive)
python src/benchmark_ogbn_distributed.py \
    --num_nodes 50 \
    --workers 2 \
    --explainers heuchase
```

**What happens:**
- Loads full OGBN-Papers100M dataset (~78GB RAM)
- Samples 50 test nodes
- Extracts L-hop subgraphs for each node
- Saves to `cache/subgraphs/subgraphs_hops2_nodes50_hash<ID>.pt`
- Runs a minimal benchmark (to verify cache works)

**Output:**
```
cache/subgraphs/subgraphs_hops2_nodes50_hash<ID>.pt
```

### Phase 2: Run from Cache (Subsequent Runs)
```bash
# Option A: Submit benchmark job with cache-only mode
sbatch run_ogbn_distributed_bench.slurm

# Option B: Run directly with cache-only flag
python src/benchmark_ogbn_distributed.py --use-cache-only
```

**What happens:**
- Skips dataset loading (saves ~78GB RAM + loading time)
- Loads subgraphs directly from cache (~2 seconds)
- Runs benchmarks on 3 explainers × 5 worker configs = 15 tests
- Saves results to `results/ogbn_distributed/`

**Memory usage:**
- Without cache-only: ~78GB (full graph) + subgraphs
- With cache-only: ~1-2GB (subgraphs only)

## Command Line Options

```bash
python src/benchmark_ogbn_distributed.py [OPTIONS]

Options:
  --config CONFIG         Path to config file (default: config.yaml)
  --use-cache-only        Only load from cache, skip dataset loading
  --num_nodes N           Number of nodes to sample (default: 100)
  --workers W1 W2 ...     List of worker counts (default: 2 4 6 8 10)
  --explainers E1 E2 ...  Explainers to test (default: heuchase apxchase gnnexplainer)
  --model_path PATH       Path to trained model (default: models/OGBN_Papers100M_epoch_20.pth)
```

## Cache Management

### List cached subgraphs
```bash
python manage_cache.py list
```

### Get cache info
```bash
python manage_cache.py info cache/subgraphs/subgraphs_hops2_nodes50_hash*.pt
```

### Clean all cache
```bash
python manage_cache.py clean
```

### Clean old caches
```bash
python manage_cache.py clean-old --days 7
```

## Architecture Details

### Cache Structure
Each cache file contains:
```python
{
    'tasks': List[SubgraphTask],  # List of subgraph tasks
    'metadata': {
        'num_hops': int,
        'num_nodes': int,
        'node_ids': List[int],
        'timestamp': str,
        'total_edges': int
    }
}
```

### SubgraphTask Structure
```python
@dataclass
class SubgraphTask:
    task_id: int
    node_id: int
    subgraph_data: Data  # PyG Data object (x, edge_index, y, etc.)
    num_edges: int
```

### Cache-Only Mode Logic
1. **main()**: Checks `--use-cache-only` flag
   - If True: Skip dataset loading, set `data = None`
   - If False: Load full dataset normally

2. **run_distributed_benchmark()**: Checks `use_cache_only` parameter
   - If True: Call `load_tasks_from_cache_only()` (no Coordinator needed)
   - If False: Create Coordinator and call `create_tasks()` (may extract if cache missing)

3. **Coordinator.create_tasks()**: Handles cache hit/miss
   - Cache hit: Load from file
   - Cache miss: Extract subgraphs and save to cache

## Troubleshooting

### Error: "Cache file not found"
**Problem:** Running with `--use-cache-only` but cache doesn't exist

**Solution:** Generate cache first:
```bash
python src/benchmark_ogbn_distributed.py --num_nodes 50 --workers 2
```

### Error: Exit code 137 (OOM)
**Problem:** Out of memory during dataset loading

**Solutions:**
1. Use `--use-cache-only` after generating cache
2. Reduce `--num_nodes` during cache generation
3. Request more memory in slurm (`--mem=256G`)

### Cache not being used
**Problem:** Still loading full dataset even with cache present

**Cause:** Not using `--use-cache-only` flag

**Solution:**
```bash
python src/benchmark_ogbn_distributed.py --use-cache-only
```

## Performance Comparison

| Mode | Memory | Load Time | First Run | Subsequent Runs |
|------|--------|-----------|-----------|-----------------|
| Normal | ~78GB | ~300s | ~1500s | ~1500s |
| With Cache | ~78GB | ~300s | ~403s (cache write) | ~2s (cache read) |
| Cache-Only | ~2GB | ~2s | N/A | ~2s |

**Key Insight:** Cache-only mode saves **~76GB RAM** and **~298s loading time**.

## Workflow Summary

```
Step 1: Generate Cache (Once)
┌─────────────────────────────────────┐
│ Load OGBN-Papers100M (~78GB, 300s) │
│ Extract L-hop subgraphs (1097s)    │
│ Save to cache/subgraphs/*.pt (2s)  │
└─────────────────────────────────────┘
                  │
                  ▼
Step 2: Run Benchmarks (Repeatedly)
┌─────────────────────────────────────┐
│ Load from cache only (~2GB, 2s)    │
│ Run 3 explainers × 5 workers       │
│ Save results to results/           │
└─────────────────────────────────────┘
```

## Example Usage

### First time setup
```bash
# Generate cache for 100 nodes
sbatch generate_cache.slurm
# OR
python src/benchmark_ogbn_distributed.py --num_nodes 100 --workers 2
```

### Run full benchmarks (using cache)
```bash
# Edit run_ogbn_distributed_bench.slurm to use cache-only mode
sbatch run_ogbn_distributed_bench.slurm
```

### Custom experiments
```bash
# Test specific explainer with specific workers
python src/benchmark_ogbn_distributed.py \
    --use-cache-only \
    --explainers heuchase \
    --workers 2 4 8

# Different worker configurations
python src/benchmark_ogbn_distributed.py \
    --use-cache-only \
    --workers 1 2 4 8 16 32
```

## Notes

1. **Cache invalidation**: Cache is keyed by (node_ids, num_hops). Different node sets create different caches.
2. **Disk usage**: Each cache file is ~50-200MB depending on subgraph sizes.
3. **Reproducibility**: Use same random seed (42) for consistent node sampling.
4. **HPC best practice**: Generate cache in one job, run benchmarks in separate jobs.
