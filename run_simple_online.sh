#!/bin/bash

#####################################################################
# Simple Online RAG - 惰性KV Cache生成
#
# 核心逻辑：
#   1. 第一次遇到文档 → 生成并保存KV
#   2. 后续遇到文档 → 直接加载KV
#
# 使用方法：
#   bash run_simple_online.sh [样本数] [topk]
#
# 示例：
#   bash run_simple_online.sh 20 10
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./simple_online_rag.py"

# 模型配置
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
DATA_PATH="./data/result_reflect.json"

# 缓存配置
CACHE_DIR="/mnt/data3/tmp/simple_kv_cache"

# 解析参数
MAX_SAMPLES=${1:-20}    # 默认20个样本
TOPK=${2:-10}           # 默认召回10个文档
DEVICE=${3:-cuda:0}     # 默认cuda:0

# 创建缓存目录
mkdir -p "${CACHE_DIR}"

echo "=========================================="
echo "Simple Online RAG - Lazy KV Generation"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "模型: ${MODEL_PATH}"
echo "数据: ${DATA_PATH}"
echo "样本数: ${MAX_SAMPLES}"
echo "TopK: ${TOPK}"
echo "缓存目录: ${CACHE_DIR}"
echo "=========================================="
echo ""

# 显示当前缓存状态
if [ -d "${CACHE_DIR}" ]; then
    CACHE_COUNT=$(ls ${CACHE_DIR}/*_key.pt 2>/dev/null | wc -l)
    echo "已缓存文档数: ${CACHE_COUNT}"

    if [ ${CACHE_COUNT} -gt 0 ]; then
        CACHE_SIZE=$(du -sh ${CACHE_DIR} 2>/dev/null | cut -f1)
        echo "缓存大小: ${CACHE_SIZE}"
    fi
fi
echo ""

# 运行测试
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --bge_model_path "${BGE_MODEL_PATH}" \
    --cache_dir "${CACHE_DIR}" \
    --topk ${TOPK} \
    --max_samples ${MAX_SAMPLES} \
    --device ${DEVICE}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 测试完成！"
    echo ""
    echo "缓存位置: ${CACHE_DIR}"

    # 显示最终缓存状态
    FINAL_COUNT=$(ls ${CACHE_DIR}/*_key.pt 2>/dev/null | wc -l)
    echo "最终缓存文档数: ${FINAL_COUNT}"

    if [ ${FINAL_COUNT} -gt 0 ]; then
        FINAL_SIZE=$(du -sh ${CACHE_DIR} 2>/dev/null | cut -f1)
        echo "最终缓存大小: ${FINAL_SIZE}"
    fi
else
    echo "✗ 测试失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
