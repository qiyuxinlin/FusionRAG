#!/bin/bash

#####################################################################
# Online Lazy Loading - Rate Sweep Script
# 在线惰性加载模式：遍历不同的 rate 值进行测试
#
# 特性：
# - 零预处理时间（跳过offline KV生成）
# - 按需生成文档KV（第一次使用时）
# - 自动保存和复用KV cache
# - 新文档强制重算（rate=1.0）
# - 全局文档池架构：使用 doc_{global_id}_key.pt 命名
# - 跨问题复用：同一文档只保存一次，所有问题共享
#####################################################################

# 基础配置
GPUS="6"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/online_lazy_sweep_DraftModel"   # 结果保存目录
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"                         # KV cache 保存路径
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 数据配置
# IMPORTANT: 使用新的优化数据集（使用doc IDs而非完整文本）
DATA_PATH="/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json" # "./data/result_reflect_optimized.json" 2wikimqa_reflect_optimized.json
DATASET_NAME="2wikimqa" # 2wikimqa | musique


# Online Lazy Loading 核心配置
REPROCESS_METHOD="DraftModel"
PREPROCESS="false"             # 关键：不做预处理
RECALL_METHOD="online_lazy"    # 关键：使用online_lazy模式
REVERT_ROPE="true"             # RoPE调整

# OpenAI 评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# Rate 列表 - 测试不同的重算比例
RATE_LIST=(0.5)

# 可选参数
MAX_SAMPLES=""  # 留空表示测试所有样本，或设置为数字（如"5"）

USE_ENTROPY_SELECTION="true"
ENTROPY_TOP_K="4"
DRAFT_LAYER_SELECTION="entropy"

# 是否清除之前的缓存（可选）
CLEAR_CACHE_BEFORE_START="true"

#####################################################################
# 开始遍历
#####################################################################

echo "=========================================="
echo "Online Lazy Loading - Rate Sweep"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "召回方法: ${RECALL_METHOD}"
echo "Rate 列表: ${RATE_LIST[@]}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_DIR}"
echo "Results: ${RESULT_DIR}"
echo "=========================================="
echo ""

# 创建必要的目录
mkdir -p "${RESULT_DIR}"
mkdir -p "${CACHE_DIR}"

# 可选：清除之前的缓存
if [ "${CLEAR_CACHE_BEFORE_START}" == "true" ]; then
    echo "⚠ 清除旧的 KV cache..."
    rm -rf "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache"/*
    echo "✓ KV cache 已清除"
    echo ""
fi

# 显示当前缓存状态
if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
    CACHE_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" 2>/dev/null | wc -l)
    if [ ${CACHE_COUNT} -gt 0 ]; then
        echo "当前已缓存文档数: ${CACHE_COUNT}"
        CACHE_SIZE=$(du -sh "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" 2>/dev/null | cut -f1)
        echo "当前缓存大小: ${CACHE_SIZE}"
        echo ""
    fi
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

# 记录开始时间
SWEEP_START_TIME=$(date +%s)

# 遍历每个 rate
for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始测试 Rate = ${RATE}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # 记录单次测试开始时间
    TEST_START_TIME=$(date +%s)
    
    # 构建参数列表
    PYTHON_ARGS=(
        "--model_type" "${MODEL_TYPE}"
        "--model_path" "${MODEL_PATH}"
        "--model_name" "${MODEL_NAME}"
        "--data_path" "${DATA_PATH}"
        "--dataset_name" "${DATASET_NAME}"
        "--cache_path" "${CACHE_DIR}"
        "--result_path" "${RESULT_DIR}"
        "--rate" "${RATE}"
        "--preprocess" "${PREPROCESS}"
        "--recall_method" "${RECALL_METHOD}"
        "--reprocess_method" "${REPROCESS_METHOD}"
        "--revert_rope" "${REVERT_ROPE}"
        "--openai_base_url" "${OPENAI_BASE_URL}"
        "--openai_api_key" "${OPENAI_API_KEY}"
        "--openai_model" "${OPENAI_MODEL}"
    )

    # 添加可选参数
    if [ ! -z "${MAX_SAMPLES}" ]; then
        PYTHON_ARGS+=("--max_samples" "${MAX_SAMPLES}")
    fi

    # 运行测试
    ${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

    EXIT_CODE=$?

    # 计算单次测试耗时
    TEST_END_TIME=$(date +%s)
    TEST_DURATION=$((TEST_END_TIME - TEST_START_TIME))
    TEST_MINUTES=$((TEST_DURATION / 60))
    TEST_SECONDS=$((TEST_DURATION % 60))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Rate ${RATE} 测试完成 (耗时: ${TEST_MINUTES}分${TEST_SECONDS}秒)"

        # 显示当前缓存状态
        if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
            CACHE_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" 2>/dev/null | wc -l)
            if [ ${CACHE_COUNT} -gt 0 ]; then
                echo "  当前已缓存文档数: ${CACHE_COUNT}"
            fi
        fi
    else
        echo "✗ Rate ${RATE} 测试失败！退出码: ${EXIT_CODE}"
        # 遇到错误时继续还是停止（取消下面注释以在失败时停止）
        # exit ${EXIT_CODE}
    fi

    echo "=========================================="
done

# 计算总耗时
SWEEP_END_TIME=$(date +%s)
TOTAL_DURATION=$((SWEEP_END_TIME - SWEEP_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo "=========================================="
echo ""
echo "存储位置："
echo "  KV Cache: ${CACHE_DIR}"
echo "  Results: ${RESULT_DIR}"
echo ""

# 显示最终缓存状态
if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
    FINAL_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" 2>/dev/null | wc -l)
    if [ ${FINAL_COUNT} -gt 0 ]; then
        FINAL_SIZE=$(du -sh "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" 2>/dev/null | cut -f1)
        echo "最终缓存统计："
        echo "  文档数: ${FINAL_COUNT}"
        echo "  大小: ${FINAL_SIZE}"
        echo ""
    fi
fi

echo "测试的 Rate 列表: ${RATE_LIST[@]}"
echo ""
echo "提示："
echo "  - 第一次运行会按需生成KV（较慢）"
echo "  - 第二次运行会复用缓存（较快）"
echo "  - 如需重新测试，可设置 CLEAR_CACHE_BEFORE_START=\"true\""
echo ""
