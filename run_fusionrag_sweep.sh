#!/bin/bash

#####################################################################
# FusionRAG Rate Sweep Script
# 快速遍历不同的 rate 值进行测试
#####################################################################

# 基础配置
GPUS="6"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/repeat_self2"       # 结果保存目录（CSV等）
CACHE_DIR="/mnt/data3/tmp/fusionrag"                        # KV cache 保存路径
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
DATA_PATH="./data/result_reflect.json"
DATASET_NAME="musique"

# FusionRAG 方法配置
REPROCESS_METHOD="FusionRAG"  # 可修改: FusionRAG, Oracle, OracleAdaptive, etc.
TOPK="10"
PREPROCESS="true"
USE_RANDOM_RECALL="true"     # [已废弃] 请使用 RECALL_METHOD
RECALL_METHOD="repeat_self"       # 召回方法: bge, random, repeat_self, fixed_doc
RANDOM_SEED="42"              # 随机种子 (当 RECALL_METHOD=random 时生效)
FIXED_DOC_IDX="0"             # 固定文档索引 (当 RECALL_METHOD=fixed_doc 时生效)
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"
USE_MULTI_GPU="false"

# KV Calibration 参数 (NEW - 2026-01-16)
# 启用KV校准功能，类似BatchNorm：offline统计偏移，online直接校准no_preprocess的KV
ENABLE_KV_CALIBRATION="false"           # 是否启用KV校准
KV_CALIBRATION_MODE="online"            # offline: 统计偏移并保存 | online: 加载偏移并应用校准
CALIBRATION_REFERENCE_METHOD="bge"      # 参考方法（用于计算偏移的preprocess方法）
CALIBRATION_SAMPLE_RATIO="0.1"          # Offline阶段：统计偏移的样本比例
CALIBRATION_GRANULARITY="per_layer"     # 统计粒度：per_layer | per_head | per_position
CALIBRATION_AGGREGATION="mean"          # 聚合方式：mean | mean_std | weighted
CALIBRATION_KEY_LAYERS=""               # 对Key校准的层（逗号分隔，如"0,1,2,3"）。留空表示所有层
CALIBRATION_VALUE_LAYERS=""             # 对Value校准的层（逗号分隔）。留空表示所有层
CALIBRATION_AUTO_SELECT_LAYERS="false"  # 是否自动选择偏移显著的层
CALIBRATION_THRESHOLD="0.1"             # 自动选择层的L2范数阈值
CALIBRATION_STATS_PATH=""               # 偏移统计量文件路径（留空则自动生成）

# 其他参数
USE_ENTROPY_SELECTION="false"
ENTROPY_TOP_K="4"
DRAFT_LAYER_SELECTION="entropy"
VATTENTION_TOPK_RATIO="0.5"
LONG_DECODE="false"
LONG_DECODE_MAX_TOKENS="1000"

# OpenAI 评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

# Rate 列表
RATE_LIST=(0.0 0.1 0.15 0.3 0.5 0.8 0.9 1.0 ) 

# 可选参数
MAX_SAMPLES="" 
DRAFT_MODEL_PATH=""

#####################################################################
# 开始遍历
#####################################################################

echo "=========================================="
echo "FusionRAG Rate Sweep"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "Rate 列表: ${RATE_LIST[@]}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_DIR}"
echo "Results: ${RESULT_DIR}"
echo "=========================================="
echo ""

# 创建必要的目录
mkdir -p "${RESULT_DIR}"
mkdir -p "${CACHE_DIR}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

# 遍历每个 rate
for RATE in "${RATE_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始测试 Rate = ${RATE}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # 构建参数列表
    PYTHON_ARGS=(
        "--model_type" "${MODEL_TYPE}"
        "--model_path" "${MODEL_PATH}"
        "--model_name" "${MODEL_NAME}"
        "--data_path" "${DATA_PATH}"
        "--dataset_name" "${DATASET_NAME}"
        "--cache_path" "${CACHE_DIR}"
        "--result_path" "${RESULT_DIR}"
        "--bge_model_path" "${BGE_MODEL_PATH}"
        "--rate" "${RATE}"
        "--topk" "${TOPK}"
        "--preprocess" "${PREPROCESS}"
        "--use_random_recall" "${USE_RANDOM_RECALL}"
        "--recall_method" "${RECALL_METHOD}"
        "--random_seed" "${RANDOM_SEED}"
        "--fixed_doc_idx" "${FIXED_DOC_IDX}"
        "--reprocess_method" "${REPROCESS_METHOD}"
        "--revert_rope" "${REVERT_ROPE}"
        "--use_multi_gpu" "${USE_MULTI_GPU}"
        "--preprocess_scope" "${PREPROCESS_SCOPE}"
        "--openai_base_url" "${OPENAI_BASE_URL}"
        "--openai_api_key" "${OPENAI_API_KEY}"
        "--openai_model" "${OPENAI_MODEL}"
        "--use_entropy_selection" "${USE_ENTROPY_SELECTION}"
        "--entropy_top_k" "${ENTROPY_TOP_K}"
        "--draft_layer_selection" "${DRAFT_LAYER_SELECTION}"
        "--vattention_topk_ratio" "${VATTENTION_TOPK_RATIO}"
        "--long_decode" "${LONG_DECODE}"
        "--long_decode_max_tokens" "${LONG_DECODE_MAX_TOKENS}"
        "--enable_kv_calibration" "${ENABLE_KV_CALIBRATION}"
        "--kv_calibration_mode" "${KV_CALIBRATION_MODE}"
        "--calibration_reference_method" "${CALIBRATION_REFERENCE_METHOD}"
        "--calibration_sample_ratio" "${CALIBRATION_SAMPLE_RATIO}"
        "--calibration_granularity" "${CALIBRATION_GRANULARITY}"
        "--calibration_aggregation" "${CALIBRATION_AGGREGATION}"
        "--calibration_key_layers" "${CALIBRATION_KEY_LAYERS}"
        "--calibration_value_layers" "${CALIBRATION_VALUE_LAYERS}"
        "--calibration_auto_select_layers" "${CALIBRATION_AUTO_SELECT_LAYERS}"
        "--calibration_threshold" "${CALIBRATION_THRESHOLD}"
        "--calibration_stats_path" "${CALIBRATION_STATS_PATH}"
    )

    # 添加可选参数
    if [ ! -z "${DRAFT_MODEL_PATH}" ]; then
        PYTHON_ARGS+=("--draft_model_path" "${DRAFT_MODEL_PATH}")
    fi

    if [ ! -z "${MAX_SAMPLES}" ]; then
        PYTHON_ARGS+=("--max_samples" "${MAX_SAMPLES}")
    fi

    # 运行测试
    ${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Rate ${RATE} 测试完成"
    else
        echo "✗ Rate ${RATE} 测试失败！退出码: ${EXIT_CODE}"
        # 可选：遇到错误时继续还是停止
        # exit ${EXIT_CODE}  # 取消注释以在失败时停止
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
echo "  KV Cache: ${CACHE_DIR}"
echo "  Results: ${RESULT_DIR}"
echo ""
echo "测试的 Rate 列表: ${RATE_LIST[@]}"
echo ""
