#!/bin/bash

#####################################################################
# Comparison Experiment - Steering Vector vs BatchNorm vs BGE
#
# This script runs comparison experiments to evaluate different methods:
# 1. No Preprocess (baseline)
# 2. Steering Vector without manifold projection
# 3. Steering Vector with manifold projection (recommended)
# 4. BGE recall (upper bound)
#####################################################################

cd /home/shm/document/exp/FusionRAG

# ============================================================
# Configuration
# ============================================================

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

STATS_OUTPUT_DIR="./kv_stats"
RESULT_DIR="./result/comparison"
DATA_PATH="./data/result_reflect.json"

# Sample IDs for computing statistics
SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

# Test parameters
GPUS="6"
RATE_LIST=(0.0 0.3 0.5 1.0)
TOPK="10"
REPROCESS_METHOD="FusionRAG"
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"

# OpenAI configuration
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

export CUDA_VISIBLE_DEVICES=${GPUS}
mkdir -p "${STATS_OUTPUT_DIR}"
mkdir -p "${RESULT_DIR}"

# ============================================================
# Step 1: Compute Steering Vectors (if not exists)
# ============================================================

echo "=========================================="
echo "Step 1: Compute Steering Vectors"
echo "=========================================="

# Method 1: Without manifold projection
STEERING_NO_PROJ="${STATS_OUTPUT_DIR}/steering_no_projection.pt"
if [ ! -f "${STEERING_NO_PROJ}" ]; then
    echo "Computing steering vectors WITHOUT manifold projection..."
    ${PYTHON_PATH} compute_kv_distribution_stats.py \
        --cache_dir "${CACHE_DIR}" \
        --dataset "${DATASET}" \
        --model_name "${MODEL_NAME}" \
        --sample_ids ${SAMPLE_IDS} \
        --output_path "${STEERING_NO_PROJ}" \
        --no_manifold_projection

    if [ $? -ne 0 ]; then
        echo "✗ Failed to compute steering vectors (no projection)"
        exit 1
    fi
else
    echo "✓ Steering vectors (no projection) already exist: ${STEERING_NO_PROJ}"
fi

# Method 2: With manifold projection (70% variance)
STEERING_MANIFOLD="${STATS_OUTPUT_DIR}/steering_manifold_0.7.pt"
if [ ! -f "${STEERING_MANIFOLD}" ]; then
    echo ""
    echo "Computing steering vectors WITH manifold projection (70% variance)..."
    ${PYTHON_PATH} compute_kv_distribution_stats.py \
        --cache_dir "${CACHE_DIR}" \
        --dataset "${DATASET}" \
        --model_name "${MODEL_NAME}" \
        --sample_ids ${SAMPLE_IDS} \
        --output_path "${STEERING_MANIFOLD}" \
        --use_manifold_projection \
        --pca_variance_threshold 0.7

    if [ $? -ne 0 ]; then
        echo "✗ Failed to compute steering vectors (manifold projection)"
        exit 1
    fi
else
    echo "✓ Steering vectors (manifold projection) already exist: ${STEERING_MANIFOLD}"
fi

echo ""
echo "✓ All steering vectors computed successfully!"

# ============================================================
# Step 2: Run Comparison Experiments
# ============================================================

echo ""
echo "=========================================="
echo "Step 2: Run Comparison Experiments"
echo "=========================================="

# Experiment 1: No Preprocess (baseline)
echo ""
echo "=========================================="
echo "Experiment 1: No Preprocess (baseline)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET}" \
        --cache_path "${CACHE_DIR}" \
        --result_path "${RESULT_DIR}/no_preprocess" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "false" \
        --recall_method "bge" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# Experiment 2: Steering without projection
echo ""
echo "=========================================="
echo "Experiment 2: Steering without projection"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET}" \
        --cache_path "${CACHE_DIR}" \
        --result_path "${RESULT_DIR}/steering_no_proj" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "true" \
        --recall_method "no_preprocess_with_bias" \
        --kv_stats_path "${STEERING_NO_PROJ}" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# Experiment 3: Steering with manifold projection (recommended)
echo ""
echo "=========================================="
echo "Experiment 3: Steering + Manifold (recommended)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET}" \
        --cache_path "${CACHE_DIR}" \
        --result_path "${RESULT_DIR}/steering_manifold" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "true" \
        --recall_method "no_preprocess_with_bias" \
        --kv_stats_path "${STEERING_MANIFOLD}" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# Experiment 4: BGE recall (upper bound)
echo ""
echo "=========================================="
echo "Experiment 4: BGE recall (upper bound)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET}" \
        --cache_path "${CACHE_DIR}" \
        --result_path "${RESULT_DIR}/bge" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "true" \
        --recall_method "bge" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  1. No Preprocess: ${RESULT_DIR}/no_preprocess/"
echo "  2. Steering (no proj): ${RESULT_DIR}/steering_no_proj/"
echo "  3. Steering + Manifold: ${RESULT_DIR}/steering_manifold/"
echo "  4. BGE (upper bound): ${RESULT_DIR}/bge/"
echo ""
echo "Compare the results to evaluate the methods!"
