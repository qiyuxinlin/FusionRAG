#!/bin/bash

#####################################################################
# Online RAG System 测试脚本
#
# 使用方法：
#   bash run_online_rag.sh [样本数] [topk] [内存限制GB]
#
# 示例：
#   bash run_online_rag.sh 20 10 8.0
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./test_online_rag.py"

# 模型配置
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
DATA_PATH="./data/result_reflect.json"

# 缓存配置
DISK_CACHE_DIR="/mnt/data3/tmp/online_kv_cache"

# 解析参数
MAX_SAMPLES=${1:-20}      # 默认20个样本
TOPK=${2:-10}             # 默认召回10个文档
MAX_MEMORY_GB=${3:-8.0}   # 默认8GB内存
MAX_NEW_TOKENS=${4:-50}   # 默认生成50 tokens

# 创建缓存目录
mkdir -p "${DISK_CACHE_DIR}"

echo "=========================================="
echo "Online RAG System Test"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "模型: ${MODEL_PATH}"
echo "数据: ${DATA_PATH}"
echo "样本数: ${MAX_SAMPLES}"
echo "TopK: ${TOPK}"
echo "内存限制: ${MAX_MEMORY_GB} GB"
echo "磁盘缓存: ${DISK_CACHE_DIR}"
echo "=========================================="
echo ""

# 运行测试
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --bge_model_path "${BGE_MODEL_PATH}" \
    --topk ${TOPK} \
    --max_samples ${MAX_SAMPLES} \
    --max_memory_gb ${MAX_MEMORY_GB} \
    --disk_cache_dir "${DISK_CACHE_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 测试完成！"
    echo ""
    echo "缓存位置: ${DISK_CACHE_DIR}"
    echo "缓存文件数: $(ls ${DISK_CACHE_DIR}/*_key.pt 2>/dev/null | wc -l)"
else
    echo "✗ 测试失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
