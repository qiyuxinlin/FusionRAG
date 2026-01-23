#!/bin/bash
# 运行 v5 实验（完整修复后）

echo "========================================"
echo "准备运行 v5 实验（完整修复版本）"
echo "========================================"
echo ""
echo "此脚本将："
echo "  1. 清空所有 KV cache（非常重要！）"
echo "  2. 更新 sweep 脚本配置为 v5"
echo "  3. 运行完整的 rate sweep"
echo ""

# 确认
read -p "确认清空 KV cache 并运行 v5 实验？(yes/no) " confirm
if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

# 清空 KV cache
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache"
echo ""
echo "正在清空 KV cache..."
if [ -d "$CACHE_DIR" ]; then
    CACHE_COUNT=$(find "$CACHE_DIR" -name "*.pt" 2>/dev/null | wc -l)
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "  当前缓存：$CACHE_COUNT 个文件，总大小 $CACHE_SIZE"

    rm -rf "$CACHE_DIR"/*
    echo "  ✓ KV cache 已清空"
else
    echo "  缓存目录不存在，无需清空"
fi

# 备份并更新 sweep 脚本
SWEEP_SCRIPT="/home/shm/document/exp/FusionRAG/run_online_lazy_sweep.sh"
BACKUP_SCRIPT="${SWEEP_SCRIPT}.v4.bak"

echo ""
echo "正在更新 sweep 脚本..."
if [ -f "$SWEEP_SCRIPT" ]; then
    # 备份
    cp "$SWEEP_SCRIPT" "$BACKUP_SCRIPT"
    echo "  ✓ 已备份原脚本到 ${BACKUP_SCRIPT}"

    # 更新 RESULT_DIR
    sed -i 's|RESULT_DIR=.*|RESULT_DIR="/home/shm/document/exp/FusionRAG/result/online_lazy_sweep_v5"|' "$SWEEP_SCRIPT"
    echo "  ✓ 已更新结果目录为 v5"
fi

# 显示当前配置
echo ""
echo "========================================"
echo "当前配置："
echo "========================================"
grep "RESULT_DIR=" "$SWEEP_SCRIPT"
grep "CACHE_DIR=" "$SWEEP_SCRIPT"
grep "CLEAR_CACHE_BEFORE_START=" "$SWEEP_SCRIPT"
grep "RATE_LIST=" "$SWEEP_SCRIPT"

# 运行实验
echo ""
echo "========================================"
echo "开始运行 v5 实验"
echo "========================================"
echo ""
echo "预计耗时：约 2-3 小时（取决于 GPU 和样本数）"
echo "可以在另一个终端监控进度："
echo "  watch -n 30 'ls -lht /home/shm/document/exp/FusionRAG/result/online_lazy_sweep_v5/Qwen2.5-7B-Instruct/musique/nopreprocess/*.txt'"
echo ""
read -p "按 Enter 开始..."

cd /home/shm/document/exp/FusionRAG
bash "$SWEEP_SCRIPT"

echo ""
echo "========================================"
echo "v5 实验完成！"
echo "========================================"
echo ""
echo "结果位置："
echo "  /home/shm/document/exp/FusionRAG/result/online_lazy_sweep_v5"
echo ""
echo "对比命令："
echo "  python3 << 'EOF'"
echo "  import os"
echo "  # 读取 v2, v4, v5 的结果并对比"
echo "  # （详见 COMPLETE_FIX_SUMMARY.md）"
echo "  EOF"
