#!/bin/bash

#####################################################################
# KV Calibration 端到端测试
# 测试完整的offline→online流程
#####################################################################

set -e  # 遇到错误立即退出

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
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/kv_calibration_test"

# 测试参数
MAX_SAMPLES=5  # 只测试5个样本，快速验证
TOPK=10
RATE=0.3

export CUDA_VISIBLE_DEVICES=${GPUS}

echo "=========================================="
echo "KV Calibration 端到端测试"
echo "=========================================="
echo "GPU: ${GPUS}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Samples: ${MAX_SAMPLES}"
echo "=========================================="
echo ""

# ========================================
# Phase 0: 生成基础KV cache (no_preprocess 和 bge)
# ========================================
echo ""
echo "=========================================="
echo "Phase 0: 生成基础KV cache"
echo "=========================================="
echo ""

echo "Step 0.1: 生成no_preprocess的KV cache..."
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
    --enable_kv_calibration false

echo ""
echo "✓ no_preprocess KV cache generated"
echo ""

echo "Step 0.2: 生成BGE preprocess的KV cache..."
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
    --preprocess true \
    --recall_method bge \
    --max_samples ${MAX_SAMPLES} \
    --enable_kv_calibration false

echo ""
echo "✓ BGE preprocess KV cache generated"
echo ""

# ========================================
# Phase 1: Offline - 统计KV偏移
# ========================================
echo ""
echo "=========================================="
echo "Phase 1: Offline统计KV偏移"
echo "=========================================="
echo ""

# 使用kv_calibration.py的offline模式
${PYTHON_PATH} kv_calibration.py \
    --mode offline \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --num_layers 28 \
    --max_samples ${MAX_SAMPLES} \
    --reference_method "bge" \
    --granularity "per_layer" \
    --aggregation "mean" \
    --auto_select_layers "true" \
    --threshold 0.05

STATS_PATH="${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer.pt"

echo ""
echo "✓ Offline统计完成"
echo "  Stats saved to: ${STATS_PATH}"
echo ""

# ========================================
# Phase 2: 查看统计摘要
# ========================================
echo ""
echo "=========================================="
echo "Phase 2: 查看统计摘要"
echo "=========================================="
echo ""

${PYTHON_PATH} kv_calibration.py \
    --mode summary \
    --stats_path "${STATS_PATH}"

echo ""

# ========================================
# Phase 3: Online - 应用KV校准进行推理
# ========================================
echo ""
echo "=========================================="
echo "Phase 3: Online应用KV校准"
echo "=========================================="
echo ""

echo "使用KV Calibration进行推理（不启用preprocess）..."
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --dataset_name "${DATASET}" \
    --cache_path "${CACHE_DIR}" \
    --result_path "${RESULT_DIR}/kv_calibrated" \
    --bge_model_path "${BGE_MODEL_PATH}" \
    --rate ${RATE} \
    --topk ${TOPK} \
    --preprocess false \
    --max_samples ${MAX_SAMPLES} \
    --enable_kv_calibration true \
    --kv_calibration_mode online \
    --calibration_stats_path "${STATS_PATH}"

echo ""
echo "✓ Online推理完成"
echo ""

# ========================================
# 总结
# ========================================
echo ""
echo "=========================================="
echo "端到端测试完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  1. KV cache:"
echo "     ${CACHE_DIR}/${DATASET}/${MODEL_NAME}/kv_cache/"
echo ""
echo "  2. Calibration stats:"
echo "     ${STATS_PATH}"
echo "     ${STATS_PATH%.pt}.json"
echo ""
echo "  3. Results:"
echo "     No preprocess:     ${RESULT_DIR}/${MODEL_NAME}/${DATASET}/nopreprocess/"
echo "     BGE preprocess:    ${RESULT_DIR}/${MODEL_NAME}/${DATASET}/FusionRAG_global_topk${TOPK}_bge/"
echo "     KV Calibrated:     ${RESULT_DIR}/kv_calibrated/${MODEL_NAME}/${DATASET}/nopreprocess/"
echo ""
echo "下一步："
echo "1. 对比三种方法的准确率（查看各自目录下的CSV文件）"
echo "2. 如果效果好，可以增大MAX_SAMPLES进行完整测试"
echo "3. 尝试不同的granularity (per_head) 或 层选择策略"
echo ""
echo "=========================================="
