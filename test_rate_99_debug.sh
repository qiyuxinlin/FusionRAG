#!/bin/bash

#####################################################################
# Quick Debug Test - Rate 0.99
# 只测试 1 个样本，查看详细的 token 选择信息
#####################################################################

# 基础配置
GPUS="7"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/debug_rate_99"
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 数据配置
DATA_PATH="/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json"
DATASET_NAME="2wikimqa"

# Online Lazy Loading 配置
REPROCESS_METHOD="DraftModel"
PREPROCESS="false"
RECALL_METHOD="online_lazy"
REVERT_ROPE="true"

# OpenAI 评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# 测试参数
RATE=0.99
MAX_SAMPLES="1"  # 只测试 1 个样本

echo "=========================================="
echo "Rate 0.99 Debug Test"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "Rate: ${RATE}"
echo "样本数: ${MAX_SAMPLES}"
echo "=========================================="
echo ""

# 创建目录
mkdir -p "${RESULT_DIR}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

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
    --max_samples "${MAX_SAMPLES}" \
    2>&1 | tee "${RESULT_DIR}/debug_output_rate_${RATE}.log"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "Debug 输出已保存到:"
echo "  ${RESULT_DIR}/debug_output_rate_${RATE}.log"
echo ""
echo "查看 debug 信息:"
echo "  grep -A 50 'DEBUG: Token Selection Details' ${RESULT_DIR}/debug_output_rate_${RATE}.log"
echo ""
