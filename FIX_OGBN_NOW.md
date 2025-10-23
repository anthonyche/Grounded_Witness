# 🚨 立即修复 OGBN 训练问题

## ❓ 问题

训练脚本立即退出（exit code 1），GPU完全未使用。

## ✅ 最可能的原因

**缺少 OGB 库** - 你的 conda 环境中没有安装 `ogb` 包。

## 🔧 立即执行（3步）

### 1️⃣ 安装 OGB
在HPC登录节点执行：
```bash
module load Miniconda3
conda activate skyexp
pip install ogb
```

### 2️⃣ 验证安装
```bash
python -c "from ogb.nodeproppred import PygNodePropPredDataset; print('✓ OGB installed successfully')"
```

如果看到 "✓ OGB installed successfully"，说明安装成功。

### 3️⃣ 运行诊断脚本
```bash
python test_ogbn_environment.py
```

应该看到：
```
======================================================================
ALL TESTS PASSED ✓
======================================================================
```

## 🚀 重新提交任务

```bash
# 提交训练任务
sbatch train_ogbn_papers100m.slurm

# 监控输出
tail -f logs/ogbn_papers100m_*.out
```

## 🔍 如果还是失败

运行调试脚本查看详细错误：
```bash
sbatch debug_ogbn.slurm
tail -f logs/ogbn_debug_*.out
```

查看错误日志：
```bash
# 找到最新的错误日志
ls -lt logs/*.err | head -1

# 查看内容
cat logs/ogbn_papers100m_<job_id>.err
```

## 📊 成功的标志

训练开始后应该看到：
```
Loading ogbn-papers100M dataset...
WARNING: This is a large dataset (~60GB). Loading may take several minutes.
Dataset loaded in X.XX minutes

Dataset Statistics:
  Nodes: 111,059,956
  Edges: 1,615,685,872
  ...

Epoch 001 | Loss: 5.xxxx | Train: 0.0xxx | Val: 0.0xxx | Test: 0.0xxx
```

## 💡 其他可能需要的包

如果 OGB 安装后还有问题，可能还需要：
```bash
pip install numpy pandas scikit-learn
pip install tqdm
```

## 🆘 紧急诊断命令

一次性检查所有依赖：
```bash
python << 'PYEOF'
import sys
packages = ['torch', 'torch_geometric', 'ogb', 'numpy', 'yaml']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - MISSING!')
        sys.exit(1)
print('\nAll packages OK!')
PYEOF
```

---

**Bottom Line**: 99% 概率是缺少 `ogb` 包，执行 `pip install ogb` 即可解决。
