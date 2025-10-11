#!/bin/bash
# 监控 ExhaustChase 运行进度

LOG_FILE="exhaustchase_full_run.log"

echo "ExhaustChase 运行进度监控"
echo "=========================="
echo ""

while true; do
    clear
    echo "ExhaustChase 运行进度监控 - $(date)"
    echo "=========================="
    echo ""
    
    # 检查进程是否还在运行
    if ps aux | grep "Run_Experiment.py --run_all" | grep -v grep > /dev/null; then
        echo "✓ 进程正在运行"
        echo ""
    else
        echo "✗ 进程已停止"
        echo ""
        break
    fi
    
    # 显示日志文件大小
    if [ -f "$LOG_FILE" ]; then
        SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
        LINES=$(wc -l < "$LOG_FILE")
        echo "日志文件: $LOG_FILE"
        echo "大小: $SIZE, 行数: $LINES"
        echo ""
    fi
    
    # 显示已完成的图数量
    COMPLETED=$(grep -c "\[ExhaustChase\] Graph" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "已完成图数量: $COMPLETED / 19"
    echo ""
    
    # 显示最近的进度
    echo "最近输出:"
    echo "--------"
    tail -10 "$LOG_FILE" 2>/dev/null | grep -E "\[ExhaustChase\]|Graphs processed|====" || echo "等待输出..."
    echo ""
    
    # 估算剩余时间（如果有足够数据）
    if [ "$COMPLETED" -gt "0" ]; then
        # 获取已用时间（从日志第一行的时间戳）
        # 这里简化处理，实际可以更精确
        echo "提示: 平均每个图大约需要 4-8 秒"
        REMAINING=$((19 - COMPLETED))
        EST_TIME=$((REMAINING * 6))
        echo "预计剩余: 约 $EST_TIME 秒"
    fi
    
    echo ""
    echo "按 Ctrl+C 停止监控（不会停止后台任务）"
    
    sleep 10
done

echo ""
echo "任务已完成！查看完整结果:"
echo "  cat $LOG_FILE"
