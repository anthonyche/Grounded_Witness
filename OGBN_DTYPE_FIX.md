# 数据类型错误修复 (RuntimeError: nll_loss Float)

## 🐛 错误信息

```
RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'
```

## 🔍 原因分析

`F.nll_loss()` 要求：
- **Input**: FloatTensor (模型输出的 log probabilities)
- **Target**: **LongTensor** (类别标签，必须是整数类型)

ogbn-papers100M 数据集的标签 `batch.y` 是 **Float 类型**，需要转换为 Long。

## ✅ 修复方案

### 训练函数 (train_epoch)
```python
# 修复前
loss = F.nll_loss(out, batch.y[:batch.batch_size].squeeze(1))

# 修复后
labels = batch.y[:batch.batch_size].squeeze(1).long()  # 转换为 Long
loss = F.nll_loss(out, labels)
```

### 评估函数 (evaluate)
```python
# 添加类型确保
y_true = torch.cat(y_true_list, dim=0).long()
y_pred = torch.cat(y_pred_list, dim=0).long()
```

## 📊 数据类型说明

| 数据 | 期望类型 | 说明 |
|------|----------|------|
| 模型输出 (out) | FloatTensor | log_softmax 输出 |
| 标签 (labels) | **LongTensor** | 类别索引 (0-171) |
| 预测 (y_pred) | LongTensor | argmax 结果 |

## ✅ 修复确认

修复后重新提交任务：
```bash
sbatch train_ogbn_papers100m.slurm
```

预期：
- ✅ 训练正常开始
- ✅ Loss 正常计算（不再报错）
- ✅ Training progress bar 更新

## 🎉 成功标志

看到类似输出：
```
Training:   0%|          | 5/9432 [00:XX<XX:XX,  X.XXit/s, loss=X.XXXX]
Training:   1%|▏         | 50/9432 [00:XX<XX:XX,  X.XXit/s, loss=X.XXXX]
```

**Loss 值在减少 = 模型在学习！**

## 相关文件
- `src/Train_OGBN_HPC_MiniBatch.py` - 已修复训练和评估函数
- `OGBN_LAZY_LOADER_FIX.md` - 内存优化方案
