# TreeCycle Distributed Benchmark - 快速参考

## 一键部署

```bash
ssh your-hpc-cluster
cd /path/to/GroundingGEXP
sbatch run_treecycle_distributed_bench.slurm
```

## 监控命令

```bash
# 检查任务状态
squeue -u $USER

# 实时查看输出
tail -f logs/treecycle_bench_*.out

# 查看最后50行
tail -50 logs/treecycle_bench_*.out

# 检查是否有错误
tail -50 logs/treecycle_bench_*.err
```

## 关键配置

| 参数 | 值 | 说明 |
|------|------|------|
| Workers | 20 | 并行进程数 |
| Target Nodes | 100 | 采样目标节点数 |
| Hops | 2 | 子图跳数 |
| Timeout | 1800s (30min) | 单任务超时 |
| Walltime | 24h | SLURM 最大运行时间 |
| Memory | 128GB | 总内存 (~6GB/worker) |
| CPUs | 20 | CPU核心数 |
| GPU | 1 | GPU数量 |

## 5个 Explainer

| # | Name | Status | Expected Behavior |
|---|------|--------|-------------------|
| 1 | HeuChase | ✅ | ~95% success, 30-60min |
| 2 | ApxChase | ✅ | ~92% success, 30-60min |
| 3 | ExhaustChase | ✅ | ~67% success, 1-3h |
| 4 | GNNExplainer | ✅ | 100% success, 5-15min |
| 5 | PGExplainer | ⚠️ | 100% timeout (skipped) |

## 输出解读

### 成功输出示例
```
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

Successful task times:
  Mean: 18.392s
  Median: 15.234s
  Min/Max: 3.456s / 1234.567s
```

### 对比输出示例
```
Progress Summary
======================================================================
HeuChase             | Makespan: 1847.23s | Success:  95/100 | Timeout:   3
ApxChase             | Makespan: 2134.56s | Success:  92/100 | Timeout:   5
ExhaustChase         | Makespan: 3456.78s | Success:  67/100 | Timeout:  30
GNNExplainer         | Makespan:  234.56s | Success: 100/100 | Timeout:   0
PGExplainer          | Makespan:    0.00s | Success:   0/100 | Timeout: 100
```

## 超时机制

```python
# 每个任务自动应用
signal.alarm(1800)  # 30分钟倒计时
try:
    result = explainer._run(...)
except TimeoutException:
    # 标记为 timeout (不算 failure)
    result = {'timeout': True}
finally:
    signal.alarm(0)  # 取消定时器
```

## 结果文件

1. **results/treecycle_distributed_benchmark.json**
   - 完整的 JSON 格式结果
   - 包含所有5个 explainer 的统计

2. **logs/treecycle_bench_JOBID.out**
   - 详细的 stdout 日志
   - 每个任务的进度

3. **logs/treecycle_bench_JOBID.err**
   - stderr 错误输出
   - 通常为空 (如果成功)

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 任务卡在 Phase 3 | 检查 `ps aux \| grep python` 是否有 worker |
| OOM 错误 | 减少 workers 到 10 |
| 全部超时 | 正常,ExhaustChase 预期会超时 |
| Job 被 CANCEL | 增加 walltime 到 48h |
| 找不到数据文件 | 检查路径: `ls datasets/TreeCycle/` |
| 找不到模型文件 | 检查路径: `ls models/TreeCycle*` |

## 预期运行时间

| Phase | Time | Description |
|-------|------|-------------|
| Extraction | ~10-30s | 提取100个子图 |
| HeuChase | ~30-60min | 第1个 explainer |
| ApxChase | ~30-60min | 第2个 explainer |
| ExhaustChase | ~1-3h | 第3个 explainer (慢) |
| GNNExplainer | ~5-15min | 第4个 explainer |
| PGExplainer | <1min | 第5个 explainer (跳过) |
| **Total** | **2-5h** | 总时间 |

## 成功标志

✅ 看到这些输出说明成功:
```
Worker 0: Task 1/5 ✓ (12.34s, 42 witnesses)
Worker 1: Task 1/8 ✓ (15.67s, 38 witnesses)
...
Coordinator: Received 5 results from worker 0
...
Results saved to results/treecycle_distributed_benchmark.json
```

❌ 看到这些输出说明有问题:
```
Worker 0: Task 1/5 ✗ ERROR: ...
...
Traceback (most recent call last):
...
```

## 下载结果

```bash
# 在本地终端运行
scp your-hpc:/path/to/GroundingGEXP/results/treecycle_distributed_benchmark.json .
scp your-hpc:/path/to/GroundingGEXP/logs/treecycle_bench_*.out .
```

## 分析结果

```bash
# 查看 JSON 结构
cat results/treecycle_distributed_benchmark.json | jq keys

# 查看 HeuChase 结果
cat results/treecycle_distributed_benchmark.json | jq .HeuChase

# 统计成功率
cat results/treecycle_distributed_benchmark.json | jq '.[] | {explainer: .explainer, success: .successful_tasks, timeout: .timeout_tasks}'
```

## 文档参考

- **完整变更**: `TREECYCLE_BENCHMARK_CHANGELOG.md`
- **部署指南**: `TREECYCLE_DEPLOYMENT.md`
- **完成报告**: `TREECYCLE_COMPLETION_REPORT.md`

## 联系支持

如果遇到无法解决的问题:
1. 保存 error log: `logs/treecycle_bench_*.err`
2. 保存最后100行输出: `tail -100 logs/treecycle_bench_*.out > debug.txt`
3. 运行预检测试: `python test_treecycle_timeout.py > preflight.txt 2>&1`
4. 提供以上文件

---

**准备好了!** 运行: `sbatch run_treecycle_distributed_bench.slurm` 🚀
