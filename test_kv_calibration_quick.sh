#!/bin/bash

#####################################################################
# KV Calibration 快速测试 (Online Only)
# 假设已经完成了offline统计，直接测试online应用
#####################################################################

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
cd /home/shm/document/exp/FusionRAG

# 配置
GPUS="5,6"
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"
DATA_PATH="./data/result_reflect.json"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/kv_calibration_quick"

# 测试参数
MAX_SAMPLES=3  # 只测试3个样本
TOPK=10
RATE=0.3

# Calibration配置
STATS_PATH="${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer.pt"
KEY_LAYERS=""        # 留空=所有层，或指定如 "0,1,2,3,4,5,6,7,8,9"
VALUE_LAYERS=""

export CUDA_VISIBLE_DEVICES=${GPUS}

echo "=========================================="
echo "KV Calibration 快速测试 (Online Mode)"
echo "=========================================="
echo "GPU: ${GPUS}"
echo "Samples: ${MAX_SAMPLES}"
echo "Stats: ${STATS_PATH}"
echo "=========================================="
echo ""

# 检查stats文件是否存在
if [ ! -f "${STATS_PATH}" ]; then
    echo "⚠️  Error: Calibration stats not found!"
    echo "   ${STATS_PATH}"
    echo ""
    echo "Please run offline mode first:"
    echo "   bash run_kv_calibration_offline.sh"
    echo "   或"
    echo "   bash test_kv_calibration_e2e.sh"
    exit 1
fi

echo "✓ Found calibration stats"
echo ""

# 查看统计摘要
echo "=========================================="
echo "Calibration Stats Summary"
echo "=========================================="
${PYTHON_PATH} kv_calibration.py \
    --mode summary \
    --stats_path "${STATS_PATH}"
echo ""

# Online推理
echo "=========================================="
echo "Running Online Inference with KV Calibration"
echo "=========================================="
echo ""

${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --dataset_name "${DATASET}" \
    --cache_path "${CACHE_DIR}" \
    --result_path "${RESULT_DIR}" \
    --bge_model_path "${BGE_MODEL_PATH}" \
    --rate ${RATE} \
    --topk ${TOPK} \
    --preprocess false \
    --max_samples ${MAX_SAMPLES} \
    --enable_kv_calibration true \
    --kv_calibration_mode online \
    --calibration_stats_path "${STATS_PATH}" \
    --calibration_key_layers "${KEY_LAYERS}" \
    --calibration_value_layers "${VALUE_LAYERS}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 测试完成！"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  ${RESULT_DIR}/${MODEL_NAME}/${DATASET}/nopreprocess/"
    echo ""
else
    echo ""
    echo "✗ 测试失败！退出码: ${EXIT_CODE}"
fi
