#!/bin/bash

#####################################################################
# Test Simple Online RAG (Integrated Version)
#
# 使用修改后的 test_fusionrag_reflect.py
# 支持惰性KV Cache生成（Simple Online模式）
#
# 使用方法：
#   bash run_simple_online_integrated.sh [样本数] [topk] [rate]
#
# 示例：
#   bash run_simple_online_integrated.sh 20 10 0.3
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
GPUS="4"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./test_fusionrag_reflect.py"

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
DATA_PATH="./data/result_reflect.json"
DATASET_NAME="musique"

# 缓存配置 (使用独立的simple_online缓存目录)
CACHE_DIR="/mnt/data3/tmp/fusionrag_simple_online"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/simple_online"

# FusionRAG参数
REPROCESS_METHOD="FusionRAG"
PREPROCESS="true"
USE_RANDOM_RECALL="false"
RECALL_METHOD="bge"  # bge | random | repeat_self | fixed_doc
RANDOM_SEED="42"
FIXED_DOC_IDX="2"
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"
USE_MULTI_GPU="false"

# OpenAI评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# 解析参数
MAX_SAMPLES=${1:-20}    # 默认20个样本
TOPK=${2:-10}           # 默认召回10个文档
RATE=${3:-0.3}          # 默认rate=0.3

# 创建必要的目录
mkdir -p "${CACHE_DIR}"
mkdir -p "${RESULT_DIR}"

# 设置GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

echo "=========================================="
echo "Simple Online RAG Test (Integrated)"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "召回方法: ${RECALL_METHOD}"
echo "TopK: ${TOPK}"
echo "Rate: ${RATE}"
echo "样本数: ${MAX_SAMPLES}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_DIR}"
echo "Results: ${RESULT_DIR}"
echo "=========================================="
echo ""

# 显示当前缓存状态
if [ -d "${CACHE_DIR}" ]; then
    HASH_CACHE_DIR="${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache/by_hash"
    if [ -d "${HASH_CACHE_DIR}" ]; then
        CACHE_COUNT=$(ls ${HASH_CACHE_DIR}/*_key.pt 2>/dev/null | wc -l)
        echo "已缓存文档数(by_hash): ${CACHE_COUNT}"

        if [ ${CACHE_COUNT} -gt 0 ]; then
            CACHE_SIZE=$(du -sh ${HASH_CACHE_DIR} 2>/dev/null | cut -f1)
            echo "Hash缓存大小: ${CACHE_SIZE}"
        fi
    fi
fi
echo ""

# 运行测试
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_type "${MODEL_TYPE}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --cache_path "${CACHE_DIR}" \
    --result_path "${RESULT_DIR}" \
    --bge_model_path "${BGE_MODEL_PATH}" \
    --rate "${RATE}" \
    --topk "${TOPK}" \
    --preprocess "${PREPROCESS}" \
    --use_random_recall "${USE_RANDOM_RECALL}" \
    --recall_method "${RECALL_METHOD}" \
    --random_seed "${RANDOM_SEED}" \
    --fixed_doc_idx "${FIXED_DOC_IDX}" \
    --reprocess_method "${REPROCESS_METHOD}" \
    --revert_rope "${REVERT_ROPE}" \
    --use_multi_gpu "${USE_MULTI_GPU}" \
    --preprocess_scope "${PREPROCESS_SCOPE}" \
    --openai_base_url "${OPENAI_BASE_URL}" \
    --openai_api_key "${OPENAI_API_KEY}" \
    --openai_model "${OPENAI_MODEL}" \
    --max_samples "${MAX_SAMPLES}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 测试完成！"
    echo ""
    echo "缓存位置: ${CACHE_DIR}"

    # 显示最终缓存状态
    if [ -d "${HASH_CACHE_DIR}" ]; then
        FINAL_COUNT=$(ls ${HASH_CACHE_DIR}/*_key.pt 2>/dev/null | wc -l)
        echo "最终缓存文档数: ${FINAL_COUNT}"

        if [ ${FINAL_COUNT} -gt 0 ]; then
            FINAL_SIZE=$(du -sh ${HASH_CACHE_DIR} 2>/dev/null | cut -f1)
            echo "最终Hash缓存大小: ${FINAL_SIZE}"
        fi
    fi

    echo ""
    echo "结果文件: ${RESULT_DIR}/${MODEL_NAME}/${DATASET_NAME}"
else
    echo "✗ 测试失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
