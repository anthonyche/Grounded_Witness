# 🔧 Slurm srun 冲突问题修复

## 问题描述

训练任务失败，错误信息：
```
srun: fatal: cpus-per-task set by two different environment variables 
SLURM_CPUS_PER_TASK=13 != SLURM_TRES_PER_TASK=cpu=8
Training failed with exit code 141
```

## 原因分析

这是 Slurm 环境变量冲突导致的问题：

1. **`SLURM_CPUS_PER_TASK=13`** - 可能来自父作业或系统默认设置
2. **`SLURM_TRES_PER_TASK=cpu=8`** - 来自你的 `#SBATCH --cpus-per-task=8`
3. **`srun`** 检测到这两个变量冲突，拒绝执行

## 解决方案

### ✅ 已修复：移除 srun

对于单节点、单任务作业（我们的情况），不需要使用 `srun`。

**修改前：**
```bash
yes y | srun python src/Train_OGBN_HPC.py ...
```

**修改后：**
```bash
yes y | python src/Train_OGBN_HPC.py ...
```

### 为什么这样可以？

- **单节点作业**：`#SBATCH --nodes=1 --ntasks=1`
- **资源已分配**：Slurm 已经为整个作业分配了 GPU、CPU、内存
- **直接执行**：Python 会自动使用分配的所有资源
- **简化脚本**：减少不必要的复杂性

### 什么时候需要 srun？

只在以下情况需要：
- **多节点作业**：`--nodes > 1`
- **多任务并行**：`--ntasks > 1`
- **MPI 程序**：需要进程间通信
- **任务数组**：需要精确控制每个子任务

## 替代方案（如果必须使用 srun）

如果你的 HPC 环境要求使用 `srun`，可以清除冲突的环境变量：

```bash
# 方案 A: 清除冲突变量
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

# 然后再使用 srun
yes y | srun --cpus-per-task=8 python src/Train_OGBN_HPC.py ...
```

或者：

```bash
# 方案 B: 使用 srun 但明确指定参数
yes y | srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python src/Train_OGBN_HPC.py ...
```

## 验证修复

重新提交作业：
```bash
sbatch train_ogbn_papers100m.slurm
```

应该看到正常的训练输出，而不是 srun 错误。

## Exit Code 含义

- **Exit code 141**: `SIGPIPE` 信号，通常由 `yes y |` 管道引起
  - 当下游进程（python）没有读取所有输入时，`yes` 收到 SIGPIPE
  - 在我们的情况下，是因为 `srun` 失败导致管道断开

修复后应该看到：
- **Exit code 0**: 成功
- **Exit code 1**: Python 脚本内部错误（需要查看日志）

## 测试

运行测试确保修复有效：
```bash
# 在登录节点测试（不会真正训练）
python src/Train_OGBN_HPC.py --help

# 提交实际作业
sbatch train_ogbn_papers100m.slurm
```

## 相关资源

- Slurm srun 文档: https://slurm.schedmd.com/srun.html
- Exit codes: https://slurm.schedmd.com/job_exit_code.html
