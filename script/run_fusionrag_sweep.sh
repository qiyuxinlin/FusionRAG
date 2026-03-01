#!/bin/bash

#####################################################################
# FusionRAG Rate Sweep Script
# 快速遍历不同的 rate 值进行测试
#####################################################################

# 基础配置
GPUS="7"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/generate_short_prompt_kv/no_preprocess"       # 结果保存目录（CSV等）
CACHE_DIR="/mnt/data3/tmp/fusionrag_new_prompt"                        # KV cache 保存路径
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
DATA_PATH="./data/result_musique_reflect_optimized.json" # result_musique_reflect_optimized.json (with global doc IDs) 
DATASET_NAME="musique" # 2wikimqa| musique | musique_310

# FusionRAG 方法配置
REPROCESS_METHOD="DraftModel"  # 可修改: FusionRAG, Oracle, OracleAdaptive, etc.
TOPK="10"
PREPROCESS="false"
USE_RANDOM_RECALL="false"     # [已废弃] 请使用 RECALL_METHOD
RECALL_METHOD="repeat_self"       # 召回方法: bge, random, repeat_self, fixed_doc, no_preprocess_with_bias
RANDOM_SEED="42"              # 随机种子 (当 RECALL_METHOD=random 时生效)
FIXED_DOC_IDX="0"             # 固定文档索引 (当 RECALL_METHOD=fixed_doc 时生效)

KV_STATS_PATH=""              # KV分布统计文件路径 (当 RECALL_METHOD=no_preprocess_with_bias 时必需)
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"
USE_MULTI_GPU="false"

# 其他参数
USE_ENTROPY_SELECTION="true"
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
RATE_LIST=(0.1) 

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
    )

    # 添加可选参数
    if [ ! -z "${DRAFT_MODEL_PATH}" ]; then
        PYTHON_ARGS+=("--draft_model_path" "${DRAFT_MODEL_PATH}")
    fi

    if [ ! -z "${MAX_SAMPLES}" ]; then
        PYTHON_ARGS+=("--max_samples" "${MAX_SAMPLES}")
    fi

    if [ ! -z "${KV_STATS_PATH}" ]; then
        PYTHON_ARGS+=("--kv_stats_path" "${KV_STATS_PATH}")
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