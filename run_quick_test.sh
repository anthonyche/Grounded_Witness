#!/bin/bash
# 分布式基准测试 - 快速启动脚本

echo "=========================================="
echo "分布式解释器基准测试 - 快速启动"
echo "=========================================="
echo ""

# 检查必需文件
echo "[1/4] 检查必需文件..."
files=(
    "src/benchmark_ogbn_distributed.py"
    "src/heuchase.py"
    "src/apxchase.py"
    "src/baselines.py"
    "models/OGBN_Papers100M_epoch_20.pth"
    "run_ogbn_distributed_bench.slurm"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        missing=$((missing + 1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "错误: $missing 个必需文件缺失!"
    exit 1
fi

echo ""
echo "[2/4] 运行基本集成测试..."
echo "  测试导入和基本功能 (约 1 分钟)"
echo ""
python test_distributed_explainer.py

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 基本集成测试失败!"
    echo "请检查错误信息后重试"
    exit 1
fi

echo ""
echo "[3/4] 运行快速分布式测试..."
echo "  测试 5 个节点，2 个 workers (约 5-10 分钟)"
echo ""
python test_distributed_quick.py

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 快速分布式测试失败!"
    echo "请检查错误信息后重试"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 所有测试通过!"
echo "=========================================="
echo ""
echo "[4/4] 提交完整基准测试..."
echo ""
echo "选项 1: 本地运行 (不推荐，时间较长)"
echo "  python src/benchmark_ogbn_distributed.py"
echo ""
echo "选项 2: 提交到 HPC (推荐)"
echo "  sbatch run_ogbn_distributed_bench.slurm"
echo ""
read -p "是否现在提交到 HPC? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "提交 Slurm job..."
    sbatch run_ogbn_distributed_bench.slurm
    
    echo ""
    echo "Job 已提交!"
    echo ""
    echo "监控命令:"
    echo "  squeue -u \$USER"
    echo "  tail -f slurm-*.out"
    echo ""
    echo "结果将保存到:"
    echo "  results/ogbn_distributed/"
    echo ""
    echo "可视化结果:"
    echo "  python visualize_ogbn_distributed.py"
else
    echo ""
    echo "未提交 job。稍后可以手动运行:"
    echo "  sbatch run_ogbn_distributed_bench.slurm"
fi

echo ""
echo "=========================================="
echo "设置完成!"
echo "=========================================="
