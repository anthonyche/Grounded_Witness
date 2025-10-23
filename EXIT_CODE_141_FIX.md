# Exit Code 141 修复说明

## 问题
```
ERROR: Training failed with exit code 141
```

## 原因
Exit code 141 = 128 + 13 = SIGPIPE (信号13)

当使用管道 `yes y | python script.py` 时：
- `yes` 不断输出 "y"
- Python 脚本不读取任何输入（OGB 数据集下载是自动的）
- 当 Python 脚本结束或关闭 stdin 时，`yes` 收到 SIGPIPE 信号
- Bash 将这个信号转换为 exit code 141

## 解决方案

**移除不必要的 `yes y |`**

修改前：
```bash
yes y | python src/Train_OGBN_HPC_MiniBatch.py ...
```

修改后：
```bash
python src/Train_OGBN_HPC_MiniBatch.py ...
```

## 为什么 yes 不需要？

OGB 数据集处理：
1. **自动下载**：首次运行时自动下载，无需用户确认
2. **已下载检测**：如果数据已存在，直接加载
3. **不需要交互**：整个过程完全自动化

## 其他改进

同时修复了：
1. ✅ 移除 `srun`（单节点作业不需要）
2. ✅ 移除 `yes y |`（不需要用户输入）
3. ✅ 使用 mini-batch training（解决 GPU OOM）
4. ✅ 添加进度条（tqdm）

## 验证

重新提交后应该看到：
```bash
Loading ogbn-papers100M dataset...
Dataset loaded in X.XX minutes
Creating mini-batch data loaders...
Data loaders created successfully!
Starting mini-batch training...
Training: 100%|██████████| 1178/1178 [XX:XX<00:00]
Epoch 001 | Loss: X.XXXX | Train: 0.XXXX | Val: 0.XXXX | Test: 0.XXXX
```

不再有 exit code 141 错误！
