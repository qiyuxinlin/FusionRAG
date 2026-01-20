#!/bin/bash

#####################################################################
# Test ONLINE_LAZY Mode - On-demand KV Generation
#
# 功能：
# - 跳过所有offline预处理
# - 在回答问题时按需生成文档KV
# - 新文档强制rate=1重算
# - 保存生成的KV供后续复用
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
GPUS="4"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./test_fusionrag_reflect_v2.py"

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 数据配置
DATA_PATH="./data/result_reflect.json"
DATASET_NAME="musique"

# 缓存配置
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"
RESULT_DIR="./result/online_lazy"

# FusionRAG参数
REPROCESS_METHOD="FusionRAG"
PREPROCESS="false"  # 关键：不做预处理
RECALL_METHOD="online_lazy"  # 关键：使用online_lazy模式
REVERT_ROPE="true"

# OpenAI评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# 解析参数
MAX_SAMPLES=${1:-5}    # 默认5个样本
RATE=${2:-0.3}         # 默认rate=0.3

# 创建必要的目录
mkdir -p "${CACHE_DIR}"
mkdir -p "${RESULT_DIR}"

# 设置GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

echo "==========================================="
echo "Online Lazy Loading Test"
echo "==========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "召回方法: ${RECALL_METHOD}"
echo "Rate: ${RATE}"
echo "样本数: ${MAX_SAMPLES}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_DIR}"
echo "Results: ${RESULT_DIR}"
echo "==========================================="
echo ""

# 显示当前缓存状态
if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
    CACHE_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" | wc -l)
    echo "已缓存文档数: ${CACHE_COUNT}"
    if [ ${CACHE_COUNT} -gt 0 ]; then
        CACHE_SIZE=$(du -sh "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" 2>/dev/null | cut -f1)
        echo "缓存大小: ${CACHE_SIZE}"
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
    --rate "${RATE}" \
    --preprocess "${PREPROCESS}" \
    --recall_method "${RECALL_METHOD}" \
    --reprocess_method "${REPROCESS_METHOD}" \
    --revert_rope "${REVERT_ROPE}" \
    --openai_base_url "${OPENAI_BASE_URL}" \
    --openai_api_key "${OPENAI_API_KEY}" \
    --openai_model "${OPENAI_MODEL}" \
    --max_samples "${MAX_SAMPLES}"

EXIT_CODE=$?

echo ""
echo "==========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 测试完成！"
    echo ""
    echo "缓存位置: ${CACHE_DIR}"

    # 显示最终缓存状态
    if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
        FINAL_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" | wc -l)
        echo "最终缓存文档数: ${FINAL_COUNT}"

        if [ ${FINAL_COUNT} -gt 0 ]; then
            FINAL_SIZE=$(du -sh "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" 2>/dev/null | cut -f1)
            echo "最终缓存大小: ${FINAL_SIZE}"
        fi
    fi

    echo ""
    echo "结果文件: ${RESULT_DIR}/${MODEL_NAME}/${DATASET_NAME}"
else
    echo "✗ 测试失败！退出码: ${EXIT_CODE}"
fi
echo "==========================================="

exit ${EXIT_CODE}
