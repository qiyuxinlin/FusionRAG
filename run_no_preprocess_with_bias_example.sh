#!/bin/bash

#####################################################################
# No Preprocess with Distribution Bias - Example Script
#
# 这个脚本演示如何使用新的 no_preprocess_with_bias 方法
# 该方法类似于 BatchNorm，通过统计 BGE 和 no_preprocess 的分布差异
# 在 online 阶段对 no_preprocess 的 KV cache 应用偏置向量
#####################################################################

cd /home/shm/document/exp/FusionRAG

# ============================================================
# 步骤 1: 计算 KV 分布统计量 (Offline 阶段)
# ============================================================
# 这一步只需要运行一次，生成分布统计文件

echo "=========================================="
echo "步骤 1: 计算 KV 分布统计量"
echo "=========================================="

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"
STATS_OUTPUT_PATH="./kv_stats/musique_qwen2.5-7b_stats.pt"

# 使用前 20 个样本计算统计量（可以根据需要调整）
SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

echo "计算 Steering Vectors（使用流形投影）..."
echo "方法: Manifold Steering (from overthinking paper)"
echo "公式: h' = h + α * r_M, where r_M = P_M @ (mean(KV_bge) - mean(KV_no_prep))"
echo "使用样本: ${SAMPLE_IDS}"
echo "输出路径: ${STATS_OUTPUT_PATH}"
echo "流形投影: 启用（PCA 方差阈值 = 0.7）"

${PYTHON_PATH} compute_kv_distribution_stats.py \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --sample_ids ${SAMPLE_IDS} \
    --chunk_id 1 \
    --max_layers 28 \
    --output_path "${STATS_OUTPUT_PATH}" \
    --use_manifold_projection \
    --pca_variance_threshold 0.7

if [ $? -ne 0 ]; then
    echo "✗ 统计量计算失败！"
    exit 1
fi

echo "✓ 统计量计算完成！"
echo ""

# ============================================================
# 步骤 2: 使用统计量运行 FusionRAG 测试 (Online 阶段)
# ============================================================
# 使用计算得到的统计量对 no_preprocess KV cache 应用偏置

echo "=========================================="
echo "步骤 2: 运行 FusionRAG 测试 (使用分布偏置)"
echo "=========================================="

# 基础配置
GPUS="6"
SCRIPT_PATH="./test_fusionrag_reflect.py"
RESULT_DIR="./result/no_preprocess_with_bias"
DATA_PATH="./data/result_reflect.json"

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# FusionRAG 方法配置
REPROCESS_METHOD="FusionRAG"
TOPK="10"
PREPROCESS="true"
RECALL_METHOD="no_preprocess_with_bias"  # 使用新的方法
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"

# OpenAI 评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# Rate 列表（可以测试不同的压缩率）
RATE_LIST=(0.0 0.3 0.5 1.0)

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

# 创建结果目录
mkdir -p "${RESULT_DIR}"

echo "方法: ${RECALL_METHOD}"
echo "统计文件: ${STATS_OUTPUT_PATH}"
echo "Rate 列表: ${RATE_LIST[@]}"
echo ""

# 遍历每个 rate
for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "测试 Rate = ${RATE}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    ${PYTHON_PATH} ${SCRIPT_PATH} \
        --model_type "${MODEL_TYPE}" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET}" \
        --cache_path "${CACHE_DIR}" \
        --result_path "${RESULT_DIR}" \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --rate "${RATE}" \
        --topk "${TOPK}" \
        --preprocess "${PREPROCESS}" \
        --recall_method "${RECALL_METHOD}" \
        --kv_stats_path "${STATS_OUTPUT_PATH}" \
        --reprocess_method "${REPROCESS_METHOD}" \
        --revert_rope "${REVERT_ROPE}" \
        --preprocess_scope "${PREPROCESS_SCOPE}" \
        --openai_base_url "${OPENAI_BASE_URL}" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --openai_model "${OPENAI_MODEL}"

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Rate ${RATE} 测试完成"
    else
        echo "✗ Rate ${RATE} 测试失败！退出码: ${EXIT_CODE}"
    fi

    echo "=========================================="
done

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "存储位置："
echo "  统计文件: ${STATS_OUTPUT_PATH}"
echo "  KV Cache: ${CACHE_DIR}"
echo "  Results: ${RESULT_DIR}"
echo ""
echo "测试的 Rate 列表: ${RATE_LIST[@]}"
echo ""
