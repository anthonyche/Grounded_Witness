# Yelp Training Setup - Complete Summary

## 📦 已创建的文件

### 1. 核心训练脚本
- **`src/Train_Yelp_HPC.py`** (完整独立的训练脚本)
  - ✅ 包含所有模型定义 (GCN-1/2/3, GAT, GraphSAGE)
  - ✅ 完整的训练循环
  - ✅ 多标签分类支持
  - ✅ Early stopping
  - ✅ 进度监控和日志
  - ✅ 自动保存最佳模型
  - ✅ 结果统计和JSON输出

### 2. SLURM作业脚本
- **`train_yelp.slurm`** (GPU版本)
  - 请求1个GPU，8个CPU核心，64GB内存
  - 时间限制：24小时
  - 包含完整的环境设置和错误处理

- **`train_yelp_cpu.slurm`** (CPU版本)
  - 请求16个CPU核心，128GB内存
  - 时间限制：48小时
  - 适用于没有GPU的HPC环境

### 3. 文档
- **`HPC_TRAINING_README.md`** - 完整使用文档
- **`QUICKSTART_HPC.md`** - 快速开始指南
- **`test_yelp_training.py`** - 本地测试脚本

## 🎯 训练的模型

| 模型 | 文件名 | 层数 | 架构 |
|------|--------|------|------|
| GCN-1 | `Yelp_gcn1_model.pth` | 1 | Graph Convolutional |
| GCN-2 | `Yelp_gcn2_model.pth` | 2 | Graph Convolutional |
| GCN-3 | `Yelp_gcn_model.pth` | 3 | Graph Convolutional |
| GAT-3 | `Yelp_gat_model.pth` | 3 | Graph Attention |
| GraphSAGE-3 | `Yelp_sage_model.pth` | 3 | GraphSAGE |

## 📊 Yelp数据集

- **节点数**: 716,847
- **边数**: 13,954,819 (无向)
- **特征维度**: 300
- **标签数**: 100 (多标签分类)
- **训练集**: ~50万节点
- **验证集**: ~10万节点
- **测试集**: ~10万节点

## ⚡ 使用流程

### 在HPC上运行

```bash
# 1. 上传文件到HPC
scp -r GroundingGEXP/ username@hpc:/path/to/directory/

# 2. SSH登录HPC
ssh username@hpc

# 3. 进入项目目录
cd /path/to/directory/GroundingGEXP

# 4. 创建必要的目录
mkdir -p logs models

# 5. 编辑SLURM脚本（根据你的HPC环境）
nano train_yelp.slurm
# 修改:
#   - #SBATCH --partition=gpu  (你的GPU分区名)
#   - #SBATCH --mail-user=your@email.com
#   - module load 命令 (根据你的HPC)

# 6. 提交作业
sbatch train_yelp.slurm

# 7. 监控进度
squeue -u $USER
tail -f logs/yelp_train_*.out
```

### 本地测试（可选）

```bash
# 测试脚本是否能运行（1个epoch快速验证）
python test_yelp_training.py
```

## 📈 预期训练时间

### GPU (例如 NVIDIA V100)
- GCN-1: ~10-15分钟
- GCN-2: ~15-20分钟
- GCN-3: ~20-30分钟
- GAT-3: ~30-45分钟
- GraphSAGE-3: ~20-30分钟
- **总计: ~2-3小时**

### CPU (16核心)
- GCN-1: ~2-3小时
- GCN-2: ~3-4小时
- GCN-3: ~4-6小时
- GAT-3: ~6-10小时
- GraphSAGE-3: ~4-6小时
- **总计: ~20-30小时**

## 📁 输出结果

训练完成后，`models/`目录包含：

```
models/
├── Yelp_gcn1_model.pth              # 5个训练好的模型
├── Yelp_gcn2_model.pth
├── Yelp_gcn_model.pth
├── Yelp_gat_model.pth
├── Yelp_sage_model.pth
└── Yelp_training_results.json       # 训练统计
```

### 训练结果JSON格式

```json
[
  {
    "model": "gcn1",
    "description": "GCN 1-layer",
    "layers": 1,
    "params": 12345,
    "val_metric": 0.8234,
    "test_acc": 0.8156,
    "train_time": 456.78,
    "path": "models/Yelp_gcn1_model.pth"
  },
  ...
]
```

## 🔧 训练配置

在`src/Train_Yelp_HPC.py`的`main()`函数中：

```python
config = {
    'data_name': 'Yelp',
    'data_root': './datasets',      # 数据集存储位置
    'random_seed': 42,               # 随机种子
    'hidden_dim': 64,                # 隐藏层维度
    'num_epochs': 200,               # 训练轮数
    'learning_rate': 0.01,           # 学习率
    'weight_decay': 5e-4,            # L2正则化
    'num_target_nodes': 50,          # 目标节点数（用于后续解释）
    'patience': 50,                  # Early stopping耐心值
}
```

## ✅ 代码特性

### Train_Yelp_HPC.py特性：
- ✅ **完全独立**: 包含所有模型定义，无需额外依赖
- ✅ **多标签支持**: 正确处理Yelp的100类多标签任务
- ✅ **GPU/CPU自动检测**: 自动使用可用的最佳设备
- ✅ **Early Stopping**: 防止过拟合
- ✅ **详细日志**: 每10个epoch输出训练/验证/测试指标
- ✅ **参数统计**: 显示每个模型的参数数量
- ✅ **时间追踪**: 记录每个模型的训练时间
- ✅ **结果保存**: JSON格式保存所有统计信息
- ✅ **错误处理**: 捕获并报告训练错误

### 训练循环特性：
1. **多标签损失**: 使用Binary Cross-Entropy
2. **双重准确率**: Exact Match + Hamming Accuracy
3. **最佳模型保存**: 基于验证集性能
4. **模型命名**: 遵循现有命名规范 `{Dataset}_{model}_model.pth`

## 🔍 监控和调试

### 检查作业状态
```bash
squeue -u $USER
```

### 实时查看输出
```bash
tail -f logs/yelp_train_<JOBID>.out
```

### 检查错误
```bash
tail -f logs/yelp_train_<JOBID>.err
```

### 查看GPU使用
```bash
nvidia-smi  # 需要在GPU节点上运行
```

## 💾 下载结果

训练完成后，从HPC下载到本地：

```bash
# 从本地机器执行：
scp -r username@hpc:/path/to/GroundingGEXP/models/ ./

# 或只下载模型文件：
scp username@hpc:/path/to/GroundingGEXP/models/Yelp_*.pth ./models/

# 下载训练日志：
scp username@hpc:/path/to/GroundingGEXP/logs/yelp_train_*.out ./logs/
```

## 🐛 常见问题

### 1. 内存不足 (OOM)
**症状**: Job被kill或CUDA OOM错误
**解决**: 
- 使用CPU版本: `sbatch train_yelp_cpu.slurm`
- 增加内存请求: `#SBATCH --mem=128G`

### 2. 模块加载失败
**症状**: `ModuleNotFoundError`
**解决**:
```bash
# 在HPC登录节点安装
pip install --user torch torch-geometric pyyaml numpy
```

### 3. CUDA版本不匹配
**症状**: CUDA errors
**解决**:
- 检查可用的CUDA版本: `module avail cuda`
- 加载兼容版本: `module load cuda/11.8`

### 4. 作业超时
**症状**: Job在完成前被终止
**解决**: 增加时间限制
```bash
#SBATCH --time=48:00:00
```

### 5. 权限问题
**症状**: Cannot write to directory
**解决**:
```bash
chmod +x train_yelp.slurm
mkdir -p logs models datasets
```

## 📚 相关文档

- **HPC_TRAINING_README.md**: 详细使用说明
- **QUICKSTART_HPC.md**: 3步快速开始
- **config_yelp.yaml**: 配置文件示例

## 🎓 后续步骤

训练完成后，这些模型可用于：

1. **ApxChase解释**: 节点级别的近似约束追踪
2. **HeuChase解释**: 启发式约束追踪
3. **GNNExplainer**: 基线对比
4. **PGExplainer**: 基线对比
5. **Coverage分析**: 计算约束覆盖率

## 📞 获取帮助

1. 查看HPC文档
2. 联系HPC支持团队
3. 查看SLURM手册: `man sbatch`
4. 检查Python环境: `python --version`

---

## ✨ 总结

所有必需的文件已准备就绪！

**上传到HPC的文件:**
1. `src/Train_Yelp_HPC.py` - 主训练脚本
2. `src/utils.py` - 工具函数
3. `train_yelp.slurm` - GPU作业脚本
4. `train_yelp_cpu.slurm` - CPU作业脚本（备用）

**执行步骤:**
1. 上传文件到HPC
2. 修改SLURM脚本中的partition和email
3. 运行 `sbatch train_yelp.slurm`
4. 等待2-3小时（GPU）或20-30小时（CPU）
5. 下载 `models/` 目录中的5个.pth文件

**预期输出:**
- 5个训练好的模型文件
- 1个JSON统计文件
- 完整的训练日志

祝训练顺利！🚀
