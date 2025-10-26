# TreeCycle Distributed Benchmark - Deployment Guide

## Quick Start

### Prerequisites (已完成 ✅)

1. **Data Generated**: `datasets/TreeCycle/treecycle_d5_bf15_n813616.pt` (813,616 nodes)
2. **Model Trained**: `models/TreeCycle_gcn_d5_bf15_n813616.pth` (2-layer GCN)
3. **Constraints Defined**: 5 TreeCycle TGDs in `src/constraints.py`
4. **Explainers Ready**: HeuChase, ApxChase, ExhaustChase, GNNExplainer, PGExplainer

### Files Updated

| File | Status | Changes |
|------|--------|---------|
| `benchmark_treecycle_distributed.py` | ✅ Updated | Added ExhaustChase, PGExplainer, 30-min timeout |
| `run_treecycle_distributed_bench.slurm` | ✅ Updated | 24-hour walltime, 5 explainers |
| `TREECYCLE_BENCHMARK_CHANGELOG.md` | ✅ Created | Complete documentation |
| `test_treecycle_timeout.py` | ✅ Created | Pre-flight tests |

### Deployment Steps

#### Step 1: Verify Local Tests (Optional, on HPC login node)

```bash
# Connect to HPC
ssh your-hpc-cluster

# Navigate to project
cd /path/to/GroundingGEXP

# Run pre-flight checks (quick, <1 minute)
python test_treecycle_timeout.py
```

**Expected**: All imports ✓, timeout mechanism ✓, data/model loading ✓

#### Step 2: Submit Job

```bash
# Submit SLURM job
sbatch run_treecycle_distributed_bench.slurm
```

**Expected Output**:
```
Submitted batch job 1234567
```

#### Step 3: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch output (updates every 2s)
watch -n 2 tail -50 logs/treecycle_bench_1234567.out

# Or follow live
tail -f logs/treecycle_bench_1234567.out
```

**Expected Progress**:
```
======================================================================
TreeCycle Distributed Benchmark
======================================================================

Loading TreeCycle graph...
Loaded: 813616 nodes, 1626464 edges

Loading GCN model...
Model loaded

Loading TreeCycle constraints...
Loaded 5 constraints

======================================================================
Distributed Benchmark: HeuChase
======================================================================
...
Worker 0: Task 1/5 ✓ (12.34s, 42 witnesses)
...

Results Summary
======================================================================
Total tasks: 100
Successful: 95
Timeout (>30min): 3
Failed: 2

Timing:
  Extraction time: 12.45s
  Execution time (makespan): 1847.23s
  Total time: 1859.68s
```

#### Step 4: Check Results

```bash
# After job completes, check results file
cat results/treecycle_distributed_benchmark.json | jq .
```

### Expected Runtime

| Explainer | Est. Makespan | Expected Timeouts |
|-----------|---------------|-------------------|
| HeuChase | 30-60 min | Low (0-5%) |
| ApxChase | 30-60 min | Low (0-5%) |
| ExhaustChase | 1-3 hours | Medium (20-40%) |
| GNNExplainer | 5-15 min | Very Low (0%) |
| PGExplainer | <1 min | High (100% - skipped) |
| **Total** | **2-5 hours** | N/A |

### Resource Usage

- **CPUs**: 20 cores (1 per worker)
- **Memory**: ~120GB peak (6GB per worker)
- **GPU**: 1 GPU (for model inference)
- **Disk**: <1GB for results
- **Walltime**: Up to 24 hours (buffer for slow tasks)

### Troubleshooting

#### Problem: Worker hangs at "Phase 3: Parallel Execution"
**Solution**: Check if multiprocessing is working:
```bash
# Check if workers are spawned
ps aux | grep python | grep worker
```

#### Problem: OOM (Out of Memory) errors
**Solution**: 
1. Reduce `num_workers` from 20 to 10 in `benchmark_treecycle_distributed.py:562`
2. Resubmit job

#### Problem: All ExhaustChase tasks timeout
**Solution**: This is expected! ExhaustChase does exhaustive enforcement which is very slow. Timeouts mean "不scale" which is a valid result.

#### Problem: Job killed with "CANCELLED" status
**Solution**: Check walltime, may need >24 hours. Update slurm script:
```bash
#SBATCH --time=48:00:00
```

### Interpreting Results

#### Success Metrics
- **Successful tasks**: Completed within 30 minutes
- **Makespan**: Parallel execution time (wall-clock)
- **Avg witnesses**: Quality of explanations

#### Scalability Assessment
- **Method scales**: <10% timeout rate
- **Method does not scale**: >30% timeout rate
- **Method failed**: High error rate (>10%)

#### Example Analysis
```
HeuChase             | Makespan: 1847s | Success:  95/100 | Timeout:   3
ApxChase             | Makespan: 2134s | Success:  92/100 | Timeout:   5
ExhaustChase         | Makespan: 3456s | Success:  67/100 | Timeout:  30
GNNExplainer         | Makespan:  234s | Success: 100/100 | Timeout:   0
PGExplainer          | Makespan:    0s | Success:   0/100 | Timeout: 100
```

**Interpretation**:
- ✅ **HeuChase**: Scales well (95% success, 31 min makespan)
- ✅ **ApxChase**: Scales well (92% success, 36 min makespan)
- ⚠️ **ExhaustChase**: Partially scales (67% success, 58 min makespan)
- ✅ **GNNExplainer**: Scales very well (100% success, 4 min makespan)
- ❌ **PGExplainer**: Does not scale (needs training, not suitable for distributed)

### Next Steps After Results

1. **Analyze JSON output**:
   ```bash
   python analyze_treecycle_results.py results/treecycle_distributed_benchmark.json
   ```

2. **Generate comparison plots**:
   - Makespan comparison (Figure 13 style)
   - Timeout rate by explainer
   - Witness count distribution

3. **Attempt larger scale** (if successful):
   - Generate depth=6, bf=20 graph (~100M nodes)
   - Update data path in benchmark script
   - Resubmit job

4. **Write up results**:
   - Which methods scale to 813K nodes?
   - What is the performance trade-off?
   - How do timeouts affect scalability assessment?

### Files to Collect

After completion, download:
- `results/treecycle_distributed_benchmark.json` (main results)
- `logs/treecycle_bench_*.out` (stdout log)
- `logs/treecycle_bench_*.err` (stderr log, if any errors)

### Support

If issues persist:
1. Check error log: `logs/treecycle_bench_*.err`
2. Verify data/model files exist
3. Check Python environment: `which python`, `python --version`
4. Test timeout mechanism: `python test_treecycle_timeout.py`

---

**Ready to Deploy!** 🚀

Run: `sbatch run_treecycle_distributed_bench.slurm`
