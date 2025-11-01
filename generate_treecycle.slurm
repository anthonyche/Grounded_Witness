#!/bin/bash
#SBATCH --job-name=treecycle_gen
#SBATCH --output=logs/treecycle_gen_%j.out
#SBATCH --error=logs/treecycle_gen_%j.err
#SBATCH --time=06:00:00            # 6 hours (中等规模足够)
#SBATCH --partition=gpu            # CPU partition (不需要 GPU)
#SBATCH --cpus-per-task=20         # 20 CPU cores
#SBATCH --mem=128G                  # 64GB RAM (中等规模足够)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # Request 1 GPU for model inference

echo "=================================================="
echo "SLURM Job: TreeCycle Graph Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Load environment
echo "Loading environment..."
module load Miniconda3
source activate skyexp

echo "Python version:"
python --version
echo ""

# Print system info
echo "System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Create output directories
mkdir -p datasets/TreeCycle
mkdir -p logs

# ================================================
# 配置参数
# ================================================
# 从小到大，逐步测试

# ============ 小规模配置（快速测试）============
# 配置 1: 小规模-1 (~54K 节点，1.1M 边)
# DEPTH=4
# BRANCHING_FACTOR=15
# CYCLE_PROB=0.2
# NUM_TYPES=5
# DESCRIPTION="Small scale (~54K nodes, ~1.1M edges) - Quick test"
# 资源要求: --mem=16G --time=00:30:00

# 配置 2: 小规模-2 (~111K 节点，2.3M 边)
# DEPTH=5
# BRANCHING_FACTOR=10
# CYCLE_PROB=0.2
# NUM_TYPES=5
# DESCRIPTION="Small-medium scale (~111K nodes, ~2.3M edges) - Quick test"
# 资源要求: --mem=32G --time=00:30:00

# ============ 中等规模配置（标准实验）============
# 配置 3: 当前配置 (~813K 节点，17M 边) ✅ 已成功生成
# DEPTH=5
# BRANCHING_FACTOR=15
# CYCLE_PROB=0.2
# NUM_TYPES=5
# DESCRIPTION="Medium scale (~813K nodes, ~17M edges) - Standard experiments"
# 资源要求: --mem=64G --time=01:00:00

# ============ 大规模配置（Scalability 测试）============
# 配置 4: 大规模-1 (~12M 节点，256M 边) ← 推荐下一步
DEPTH=6
BRANCHING_FACTOR=15
CYCLE_PROB=0.15
NUM_TYPES=5
DESCRIPTION="Large scale (~12M nodes, ~256M edges) - Scalability test"
# 资源要求: --mem=128G --time=02:00:00

# 配置 5: 大规模-2 (~67M 节点，1.4B 边)
# DEPTH=6
# BRANCHING_FACTOR=20
# CYCLE_PROB=0.15
# NUM_TYPES=5
# DESCRIPTION="Very large scale (~67M nodes, ~1.4B edges) - Scalability test"
# 资源要求: --mem=256G --time=04:00:00
# 注意：需要修改 Slurm 头部：
#   #SBATCH --mem=256G
#   #SBATCH --time=04:00:00

# 配置 6: 超大规模 (~254M 节点，5.3B 边)
# DEPTH=6
# BRANCHING_FACTOR=25
# CYCLE_PROB=0.1
# NUM_TYPES=5
# DESCRIPTION="Ultra large scale (~254M nodes, ~5.3B edges) - Extreme test"
# 资源要求: --mem=512G --time=08:00:00 --partition=bigmem
# 注意：需要修改 Slurm 头部：
#   #SBATCH --mem=512G
#   #SBATCH --time=08:00:00
#   #SBATCH --partition=bigmem  # 如果 HPC 有这个 partition

echo "=================================================="
echo "TreeCycle Generation Configuration"
echo "=================================================="
echo "Description: $DESCRIPTION"
echo "  Depth: $DEPTH"
echo "  Branching factor: $BRANCHING_FACTOR"
echo "  Cycle probability: $CYCLE_PROB"
echo "  Node types: $NUM_TYPES"
echo "=================================================="
echo ""

# Calculate expected nodes (approximate)
if [ $BRANCHING_FACTOR -eq 1 ]; then
    EXPECTED_NODES=$((DEPTH + 1))
else
    # N ≈ (b^(d+1) - 1) / (b - 1)
    POWER=$((DEPTH + 1))
    EXPECTED_NODES=$(python3 -c "print(($BRANCHING_FACTOR ** $POWER - 1) // ($BRANCHING_FACTOR - 1))")
fi

echo "Expected nodes: $(printf "%'d" $EXPECTED_NODES)"
echo "Estimated memory: $(python3 -c "print(f'{$EXPECTED_NODES * 50 / 1e9:.1f} GB')")"
echo "Estimated time: $(python3 -c "
if $EXPECTED_NODES < 1000000:
    print('< 5 minutes')
elif $EXPECTED_NODES < 10000000:
    print('10-30 minutes')
elif $EXPECTED_NODES < 100000000:
    print('1-2 hours')
else:
    print('3-6 hours')
")"
echo ""

# Run generation
echo "=================================================="
echo "Starting Graph Generation"
echo "=================================================="
echo ""

START_TIME=$(date +%s)

python TreeCycleGenerator.py \
    --depth $DEPTH \
    --branching-factor $BRANCHING_FACTOR \
    --cycle-prob $CYCLE_PROB \
    --num-types $NUM_TYPES \
    --output-dir datasets/TreeCycle \
    --seed 42

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Graph generation completed successfully!"
    echo "Elapsed time: $(($ELAPSED / 60)) minutes $(($ELAPSED % 60)) seconds"
else
    echo "✗ ERROR: Graph generation failed with exit code $EXIT_CODE"
fi
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="
echo ""

# Print generated files
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generated files:"
    ls -lh datasets/TreeCycle/treecycle_d${DEPTH}_bf${BRANCHING_FACTOR}_*.pt 2>/dev/null || echo "  No files found"
    echo ""
    
    # Print file size
    if ls datasets/TreeCycle/treecycle_d${DEPTH}_bf${BRANCHING_FACTOR}_*.pt 1> /dev/null 2>&1; then
        FILE=$(ls datasets/TreeCycle/treecycle_d${DEPTH}_bf${BRANCHING_FACTOR}_*.pt | head -1)
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "Graph file size: $SIZE"
    fi
fi

echo ""
echo "=================================================="
echo "Next Steps"
echo "=================================================="
echo "1. Check the generated graph:"
echo "   cat logs/treecycle_gen_$SLURM_JOB_ID.out"
echo ""
echo "2. If successful, train a GNN model:"
echo "   sbatch train_treecycle.slurm"
echo ""
echo "3. Run witness generation:"
echo "   sbatch benchmark_treecycle.slurm"
echo ""
echo "4. For larger graphs, modify this script:"
echo "   - Uncomment 配置 2/3/4"
echo "   - Increase --mem and --time in SBATCH directives"
echo "=================================================="

exit $EXIT_CODE
