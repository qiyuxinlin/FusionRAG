#!/bin/bash

#####################################################################
# Test Rate 0.99 - After Fix (No Prefix Cache Special Case)
# 测试修改后的行为：所有文档都参与 rate-based 选择
#####################################################################

GPUS="7"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/debug_rate_99_fixed"
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"
cd /home/shm/document/exp/FusionRAG

MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

DATA_PATH="/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json"
DATASET_NAME="2wikimqa"

REPROCESS_METHOD="DraftModel"
PREPROCESS="false"
RECALL_METHOD="online_lazy"
REVERT_ROPE="true"

OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

RATE=0.99
MAX_SAMPLES="1"

echo "=========================================="
echo "Rate 0.99 Test - Fixed (All Docs Participate)"
echo "=========================================="
echo "修改: 所有文档都参与 rate-based importance 选择"
echo "没有 prefix cache 特例"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

mkdir -p "${RESULT_DIR}"
export CUDA_VISIBLE_DEVICES=${GPUS}

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
    2>&1 | tee "${RESULT_DIR}/output_rate_${RATE}.log"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "输出日志: ${RESULT_DIR}/output_rate_${RATE}.log"
echo ""
echo "查看 debug 信息:"
echo "  grep -A 40 'DEBUG: Token Selection Details' ${RESULT_DIR}/output_rate_${RATE}.log"
echo ""
