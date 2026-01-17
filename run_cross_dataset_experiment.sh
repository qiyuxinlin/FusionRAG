#!/bin/bash

#####################################################################
# Cross-Dataset Experiment: 2wikimqa (train) → musique (test)
#
# Offline: Compute steering vectors on 2wikimqa dataset
# Online: Test on musique dataset
#
# This tests the generalization ability of steering vectors
#####################################################################

cd /home/shm/document/exp/FusionRAG

# ============================================================
# Configuration
# ============================================================

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"

# Paths for 2wikimqa (for computing statistics)
TRAIN_CACHE_BASE="/mnt/data3/tmp/fusionrag"  # Base directory
TRAIN_DATASET="2wikimqa"

# Paths for musique (for testing)
TEST_CACHE_BASE="/mnt/data3/tmp/fusionrag"  # Base directory
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"

# Model paths
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# Output paths
STATS_OUTPUT_DIR="./kv_stats/cross_dataset"
RESULT_DIR="./result/cross_dataset"

# Sample selection for computing statistics (2wikimqa)
# Method 1: Manually specify sample IDs
USE_ALL_SAMPLES="false"  # Set to "true" to auto-scan all samples, "false" to use SAMPLE_IDS
SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

# Test parameters
GPUS="6"
RATE_LIST=(0.0 0.3 0.5 1.0)
TOPK="10"
STEERING_ALPHA="1.0"  # Steering vector strength (α): 1.0 = full steering, <1.0 = weaker, >1.0 = stronger
STEERING_KEY_LAYERS="all"  # Layers to apply key steering: "all", "0-10", "0,5,10", or "0-10,15,20-25"
STEERING_VALUE_LAYERS="all"  # Layers to apply value steering: "all", "0-10", "0,5,10", or "0-10,15,20-25"
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
# Step 1: Compute Steering Vectors on 2wikimqa
# ============================================================

echo "=========================================="
echo "Step 1: Compute Steering Vectors on 2wikimqa"
echo "=========================================="
echo "Training dataset: ${TRAIN_DATASET}"
echo "Cache base: ${TRAIN_CACHE_BASE}"
echo "Model: ${MODEL_NAME}"
echo "Sample IDs: ${SAMPLE_IDS}"
echo ""

# Method 1: Without manifold projection
STEERING_NO_PROJ="${STATS_OUTPUT_DIR}/2wikimqa_to_musique_no_proj.pt"
if [ ! -f "${STEERING_NO_PROJ}" ]; then
    echo "Computing steering vectors WITHOUT manifold projection..."

    # Build command based on USE_ALL_SAMPLES
    CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
        --cache_dir \"${TRAIN_CACHE_BASE}\" \
        --dataset \"${TRAIN_DATASET}\" \
        --model_name \"${MODEL_NAME}\""

    if [ "${USE_ALL_SAMPLES}" = "true" ]; then
        CMD="${CMD} --use_all_samples"
        echo "  Using all available samples (auto-scan)"
    else
        CMD="${CMD} --sample_ids ${SAMPLE_IDS}"
        echo "  Using specified sample IDs: ${SAMPLE_IDS}"
    fi

    CMD="${CMD} \
        --chunk_id 1 \
        --max_layers 28 \
        --output_path \"${STEERING_NO_PROJ}\" \
        --no_manifold_projection"

    eval ${CMD}

    if [ $? -ne 0 ]; then
        echo "✗ Failed to compute steering vectors (no projection)"
        exit 1
    fi
else
    echo "✓ Steering vectors (no projection) already exist: ${STEERING_NO_PROJ}"
fi

# Method 2: With manifold projection (70% variance)
STEERING_MANIFOLD="${STATS_OUTPUT_DIR}/2wikimqa_to_musique_manifold.pt"
if [ ! -f "${STEERING_MANIFOLD}" ]; then
    echo ""
    echo "Computing steering vectors WITH manifold projection (70% variance)..."

    # Build command based on USE_ALL_SAMPLES
    CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
        --cache_dir \"${TRAIN_CACHE_BASE}\" \
        --dataset \"${TRAIN_DATASET}\" \
        --model_name \"${MODEL_NAME}\""

    if [ "${USE_ALL_SAMPLES}" = "true" ]; then
        CMD="${CMD} --use_all_samples"
        echo "  Using all available samples (auto-scan)"
    else
        CMD="${CMD} --sample_ids ${SAMPLE_IDS}"
        echo "  Using specified sample IDs: ${SAMPLE_IDS}"
    fi

    CMD="${CMD} \
        --chunk_id 1 \
        --max_layers 28 \
        --output_path \"${STEERING_MANIFOLD}\" \
        --use_manifold_projection \
        --pca_variance_threshold 0.7"

    eval ${CMD}

    if [ $? -ne 0 ]; then
        echo "✗ Failed to compute steering vectors (manifold projection)"
        exit 1
    fi
else
    echo "✓ Steering vectors (manifold projection) already exist: ${STEERING_MANIFOLD}"
fi

echo ""
echo "✓ Steering vectors computed on 2wikimqa!"

# ============================================================
# Step 2: Test on musique Dataset
# ============================================================

echo ""
echo "=========================================="
echo "Step 2: Test Steering Vectors on musique"
echo "=========================================="
echo "Test dataset: ${TEST_DATASET}"
echo "Cache base: ${TEST_CACHE_BASE}"
echo ""

# Experiment 1: No Preprocess (baseline)
echo ""
echo "=========================================="
echo "Experiment 1: No Preprocess (baseline on musique)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TEST_DATA_PATH}" \
        --dataset_name "${TEST_DATASET}" \
        --cache_path "${TEST_CACHE_BASE}" \
        --result_path "${RESULT_DIR}/musique_baseline" \
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

# Experiment 2: Steering without projection (cross-dataset)
echo ""
echo "=========================================="
echo "Experiment 2: Steering w/o projection (2wikimqa → musique)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TEST_DATA_PATH}" \
        --dataset_name "${TEST_DATASET}" \
        --cache_path "${TEST_CACHE_BASE}" \
        --result_path "${RESULT_DIR}/musique_steering_no_proj" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "true" \
        --recall_method "no_preprocess_with_bias" \
        --kv_stats_path "${STEERING_NO_PROJ}" \
        --steering_alpha "${STEERING_ALPHA}" \
        --steering_key_layers "${STEERING_KEY_LAYERS}" \
        --steering_value_layers "${STEERING_VALUE_LAYERS}" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# Experiment 3: Steering with manifold projection (cross-dataset, recommended)
echo ""
echo "=========================================="
echo "Experiment 3: Steering + Manifold (2wikimqa → musique, recommended)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TEST_DATA_PATH}" \
        --dataset_name "${TEST_DATASET}" \
        --cache_path "${TEST_CACHE_BASE}" \
        --result_path "${RESULT_DIR}/musique_steering_manifold" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "true" \
        --recall_method "no_preprocess_with_bias" \
        --kv_stats_path "${STEERING_MANIFOLD}" \
        --steering_alpha "${STEERING_ALPHA}" \
        --steering_key_layers "${STEERING_KEY_LAYERS}" \
        --steering_value_layers "${STEERING_VALUE_LAYERS}" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    echo "✓ Rate ${RATE} complete"
done

# Experiment 4: BGE recall (upper bound on musique)
echo ""
echo "=========================================="
echo "Experiment 4: BGE recall (upper bound on musique)"
echo "=========================================="

for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "Testing Rate = ${RATE}"

    ${PYTHON_PATH} test_fusionrag_reflect.py \
        --model_type "qwen" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${TEST_DATA_PATH}" \
        --dataset_name "${TEST_DATASET}" \
        --cache_path "${TEST_CACHE_BASE}" \
        --result_path "${RESULT_DIR}/musique_bge" \
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
echo "Cross-Dataset Experiment Completed!"
echo "=========================================="
echo ""
echo "Training: 2wikimqa dataset"
echo "Testing: musique dataset"
echo ""
echo "Steering vectors:"
echo "  - No projection: ${STEERING_NO_PROJ}"
echo "  - With manifold: ${STEERING_MANIFOLD}"
echo ""
echo "Results saved in:"
echo "  1. Baseline (musique): ${RESULT_DIR}/musique_baseline/"
echo "  2. Steering w/o proj: ${RESULT_DIR}/musique_steering_no_proj/"
echo "  3. Steering + Manifold: ${RESULT_DIR}/musique_steering_manifold/"
echo "  4. BGE (upper bound): ${RESULT_DIR}/musique_bge/"
echo ""
echo "This tests the generalization ability of steering vectors across datasets!"
