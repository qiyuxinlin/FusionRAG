#!/bin/bash
# 清空 KV cache 并重新运行实验以验证修复

CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/online_lazy_sweep_v4"

echo "========================================"
echo "准备重新运行实验以验证修复"
echo "========================================"
echo ""
echo "警告：即将删除所有 KV cache！"
echo "Cache 目录: $CACHE_DIR"
echo "新结果目录: $RESULT_DIR"
echo ""
read -p "确认删除并重新运行？(yes/no) " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

# 检查是否有实验在运行
running=$(ps aux | grep "test_fusionrag_reflect_v2.py" | grep -v grep | wc -l)
if [ "$running" -gt 0 ]; then
    echo "错误：检测到有实验正在运行！"
    ps aux | grep "test_fusionrag_reflect_v2.py" | grep -v grep
    echo ""
    read -p "是否终止这些进程？(yes/no) " kill_confirm
    if [ "$kill_confirm" == "yes" ]; then
        pkill -f "test_fusionrag_reflect_v2.py"
        sleep 2
        echo "已终止正在运行的实验"
    else
        echo "请手动终止实验后再运行此脚本"
        exit 1
    fi
fi

# 备份当前 cache（可选）
echo ""
echo "备份当前 KV cache..."
BACKUP_DIR="${CACHE_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
mv "$CACHE_DIR" "$BACKUP_DIR" 2>/dev/null && echo "  已备份到: $BACKUP_DIR" || echo "  无需备份（目录不存在或已清空）"

# 创建新的结果目录
mkdir -p "$RESULT_DIR"

echo ""
echo "========================================"
echo "开始运行实验（这将需要较长时间）"
echo "========================================"
echo ""
echo "提示：可以在另一个终端监控进度："
echo "  watch -n 10 'ls -lh $RESULT_DIR/Qwen2.5-7B-Instruct/musique/nopreprocess/'"
echo ""

cd /home/shm/document/exp/FusionRAG
bash run_online_lazy_sweep.sh --result_dir "$RESULT_DIR"
