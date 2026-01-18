#!/bin/bash

#####################################################################
# FusionRAG Rate Sweep Script
# 快速遍历不同的 rate 值进行测试
#####################################################################

# 基础配置
GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/steering_head_0.9"       # 结果保存目录（CSV等）
CACHE_DIR="/mnt/data3/tmp/fusionrag"                        # KV cache 保存路径
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen3-32B"
MODEL_NAME="Qwen3-32B"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置
# 注意：如果使用跨数据集 steering（如 2wikimqa → musique），需要分别设置训练和测试数据集
TRAIN_DATASET="2wikimqa"              # 用于计算 steering vectors 的数据集（影响 KV cache 路径）
TEST_DATASET="musique"                # 用于实际测试的数据集
TEST_DATA_PATH="./data/result_reflect.json"  # 测试数据的 JSON 文件路径

# 向后兼容：如果想用同一数据集做训练和测试，可以设置：
# TRAIN_DATASET="2wikimqa"
# TEST_DATASET="2wikimqa"
# TEST_DATA_PATH="./data/2wikimqa_reflect.json"

# FusionRAG 方法配置
REPROCESS_METHOD="FusionRAG"  # 可修改: FusionRAG, Oracle, OracleAdaptive, etc.
TOPK="10"
PREPROCESS="true"
USE_RANDOM_RECALL="false"     # [已废弃] 请使用 RECALL_METHOD
RECALL_METHOD="no_preprocess_with_bias"  # 召回方法: bge, random, repeat_self, fixed_doc, no_preprocess_with_bias
RANDOM_SEED="42"              # 随机种子 (当 RECALL_METHOD=random 时生效)
FIXED_DOC_IDX="0"             # 固定文档索引 (当 RECALL_METHOD=fixed_doc 时生效)

# Steering Vector 配置 (当 RECALL_METHOD=no_preprocess_with_bias 时生效)
COMPUTE_STEERING="true"       # 是否在运行前计算 steering vectors
USE_ALL_SAMPLES="true"       # 是否使用所有样本计算 steering (true) 还是指定样本 (false)
# STEERING_SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"  # 样本 ID 列表
STEERING_OUTPUT_DIR="/mnt/data3/tmp/fusionrag/kv_stats"  # Steering vectors 输出目录
USE_MANIFOLD_PROJECTION="true"    # 是否使用 manifold projection
PCA_VARIANCE_THRESHOLD="0.9"      # PCA 方差阈值
PER_HEAD_STATS="true"            # 是否对每层的每个head单独统计 (false: 所有heads一起统计, true: 每个head独立)

STEERING_ALPHA_LIST=(0.2 0.3 0.4 0.5)  # Steering vector 强度系数列表（遍历）
STEERING_KEY_LAYERS="all"         # 应用 key steering 的层 ("all", "0-10", "0,5,10", 等)
STEERING_VALUE_LAYERS="all"       # 应用 value steering 的层

KV_STATS_PATH=""              # KV分布统计文件路径 (自动设置或手动指定)
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"
USE_MULTI_GPU="false"

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
RATE_LIST=(0.1 0.15 0.3 0.5 0.8 ) 

# 可选参数
MAX_SAMPLES="" 
DRAFT_MODEL_PATH=""

#####################################################################
# 开始遍历
#####################################################################

echo "=========================================="
echo "FusionRAG Multi-Parameter Sweep"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "召回方法: ${RECALL_METHOD}"
echo "Rate 列表: ${RATE_LIST[@]}"

# 显示数据集配置
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ] && [ "${COMPUTE_STEERING}" = "true" ]; then
    # 跨数据集或同数据集
    if [ "${TRAIN_DATASET}" != "${TEST_DATASET}" ]; then
        echo "数据集配置: 跨数据集 (${TRAIN_DATASET} → ${TEST_DATASET})"
        echo "  训练数据集: ${TRAIN_DATASET} (用于计算 steering vectors)"
        echo "  测试数据集: ${TEST_DATASET} (用于实际测试)"
    else
        echo "数据集: ${TEST_DATASET} (同数据集训练和测试)"
    fi
else
    echo "数据集: ${TEST_DATASET}"
fi

echo "测试数据路径: ${TEST_DATA_PATH}"
echo "KV Cache: ${CACHE_DIR}"
echo "Results: ${RESULT_DIR}"

# 如果使用 no_preprocess_with_bias 方法，显示 steering 配置
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ]; then
    echo ""
    echo "Steering Vector 配置:"
    echo "  计算 Steering: ${COMPUTE_STEERING}"
    if [ "${COMPUTE_STEERING}" = "true" ]; then
        echo "  训练数据集: ${TRAIN_DATASET}"
        echo "  使用所有样本: ${USE_ALL_SAMPLES}"
        if [ "${USE_ALL_SAMPLES}" = "false" ]; then
            echo "  样本数量: $(echo ${STEERING_SAMPLE_IDS} | wc -w)"
        fi
        echo "  Manifold 投影: ${USE_MANIFOLD_PROJECTION}"
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            echo "  PCA 阈值: ${PCA_VARIANCE_THRESHOLD}"
        fi
        echo "  Per-head 统计: ${PER_HEAD_STATS}"
    fi
    echo "  Alpha 列表: ${STEERING_ALPHA_LIST[@]}"
    echo "  Key 层: ${STEERING_KEY_LAYERS}"
    echo "  Value 层: ${STEERING_VALUE_LAYERS}"
fi

echo "=========================================="
echo ""
echo "将遍历 ${#STEERING_ALPHA_LIST[@]} 个 Alpha 值 × ${#RATE_LIST[@]} 个 Rate 值"
echo "总计: $((${#STEERING_ALPHA_LIST[@]} * ${#RATE_LIST[@]})) 次测试"
echo ""

# 创建必要的目录
mkdir -p "${RESULT_DIR}"
mkdir -p "${CACHE_DIR}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

#####################################################################
# Step 0: 计算 Steering Vectors (如果需要)
#####################################################################

if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ] && [ "${COMPUTE_STEERING}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Step 0: 计算 Steering Vectors"
    echo "=========================================="
    echo "训练数据集: ${TRAIN_DATASET}"
    echo "模型: ${MODEL_NAME}"

    # 创建输出目录
    mkdir -p "${STEERING_OUTPUT_DIR}"

    # 设置输出文件路径
    if [ -z "${KV_STATS_PATH}" ]; then
        # 自动生成文件名
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            SUFFIX="manifold"
            # 添加PCA阈值到文件名（去掉小数点，如0.7→07，0.95→095）
            PCA_SUFFIX=$(echo "${PCA_VARIANCE_THRESHOLD}" | sed 's/0\.//' | sed 's/\.//')
            SUFFIX="${SUFFIX}_pca${PCA_SUFFIX}"
        else
            SUFFIX="no_proj"
        fi

        # 添加per-head标记
        if [ "${PER_HEAD_STATS}" = "true" ]; then
            SUFFIX="${SUFFIX}_perhead"
        fi

        # 如果是跨数据集，文件名中体现训练→测试的关系
        if [ "${TRAIN_DATASET}" != "${TEST_DATASET}" ]; then
            KV_STATS_PATH="${STEERING_OUTPUT_DIR}/${TRAIN_DATASET}_to_${TEST_DATASET}_steering_${SUFFIX}.pt"
        else
            KV_STATS_PATH="${STEERING_OUTPUT_DIR}/${TRAIN_DATASET}_steering_${SUFFIX}.pt"
        fi
    fi

    echo "输出路径: ${KV_STATS_PATH}"
    if [ "${TRAIN_DATASET}" != "${TEST_DATASET}" ]; then
        echo "跨数据集模式: ${TRAIN_DATASET} → ${TEST_DATASET}"
    fi

    # 检查文件是否已存在
    if [ -f "${KV_STATS_PATH}" ]; then
        echo ""
        echo "✓ Steering vectors 文件已存在: ${KV_STATS_PATH}"
        echo "  跳过计算。如需重新计算，请删除此文件。"
    else
        echo ""
        echo "开始计算 steering vectors..."

        # 构建命令
        CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
            --cache_dir \"${CACHE_DIR}\" \
            --dataset \"${TRAIN_DATASET}\" \
            --model_name \"${MODEL_NAME}\""

        # 添加样本选择参数
        if [ "${USE_ALL_SAMPLES}" = "true" ]; then
            CMD="${CMD} --use_all_samples"
            echo "  使用所有可用样本 (auto-scan)"
        else
            CMD="${CMD} --sample_ids ${STEERING_SAMPLE_IDS}"
            echo "  使用指定样本: ${STEERING_SAMPLE_IDS}"
        fi

        # 添加其他参数
        CMD="${CMD} \
            --chunk_id 1 \
            --max_layers 28 \
            --output_path \"${KV_STATS_PATH}\" \
            --pca_variance_threshold ${PCA_VARIANCE_THRESHOLD}"

        # 添加 manifold projection 参数
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            CMD="${CMD} --use_manifold_projection"
            echo "  Manifold projection: 启用 (PCA 阈值: ${PCA_VARIANCE_THRESHOLD})"
        else
            CMD="${CMD} --no_manifold_projection"
            echo "  Manifold projection: 禁用"
        fi

        # 添加 per-head statistics 参数
        if [ "${PER_HEAD_STATS}" = "true" ]; then
            CMD="${CMD} --per_head_stats"
            echo "  Per-head statistics: 启用 (每个head单独统计)"
        else
            echo "  Per-head statistics: 禁用 (所有heads一起统计)"
        fi

        echo ""
        echo "执行命令:"
        echo "${CMD}"
        echo ""

        # 执行计算
        eval ${CMD}

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Steering vectors 计算成功！"
            echo "  保存位置: ${KV_STATS_PATH}"
        else
            echo ""
            echo "✗ Steering vectors 计算失败！"
            exit 1
        fi
    fi

    echo "=========================================="
    echo ""
fi

#####################################################################
# 遍历 Alpha 和 Rate
#####################################################################

# 遍历每个 steering alpha
for STEERING_ALPHA in "${STEERING_ALPHA_LIST[@]}"; do
    echo ""
    echo "######################################################################"
    echo "开始测试 Steering Alpha = ${STEERING_ALPHA}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"
    echo ""

    # 为每个 alpha 创建独立的结果目录
    ALPHA_RESULT_DIR="${RESULT_DIR}/alpha_${STEERING_ALPHA}"
    mkdir -p "${ALPHA_RESULT_DIR}"

    echo "当前 Alpha 结果目录: ${ALPHA_RESULT_DIR}"
    echo ""

    # 遍历每个 rate
    for RATE in "${RATE_LIST[@]}"; do
        echo ""
        echo "=========================================="
        echo "测试 Alpha = ${STEERING_ALPHA}, Rate = ${RATE}"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        # 构建参数列表
        PYTHON_ARGS=(
            "--model_type" "${MODEL_TYPE}"
            "--model_path" "${MODEL_PATH}"
            "--model_name" "${MODEL_NAME}"
            "--data_path" "${TEST_DATA_PATH}"
            "--dataset_name" "${TEST_DATASET}"
            "--cache_path" "${CACHE_DIR}"
            "--result_path" "${ALPHA_RESULT_DIR}"
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

        # Steering vector 相关参数 (当 RECALL_METHOD=no_preprocess_with_bias 时)
        if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ]; then
            if [ ! -z "${KV_STATS_PATH}" ]; then
                PYTHON_ARGS+=("--kv_stats_path" "${KV_STATS_PATH}")
            fi
            PYTHON_ARGS+=("--steering_alpha" "${STEERING_ALPHA}")
            PYTHON_ARGS+=("--steering_key_layers" "${STEERING_KEY_LAYERS}")
            PYTHON_ARGS+=("--steering_value_layers" "${STEERING_VALUE_LAYERS}")

            # 如果启用per-head统计，传递per-head参数
            if [ "${PER_HEAD_STATS}" = "true" ]; then
                PYTHON_ARGS+=("--use_per_head_steering")
            fi
        fi

        # 运行测试
        ${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

        EXIT_CODE=$?

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "✓ Alpha ${STEERING_ALPHA}, Rate ${RATE} 测试完成"
        else
            echo "✗ Alpha ${STEERING_ALPHA}, Rate ${RATE} 测试失败！退出码: ${EXIT_CODE}"
            # 可选：遇到错误时继续还是停止
            # exit ${EXIT_CODE}  # 取消注释以在失败时停止
        fi

        echo "=========================================="
    done  # Rate loop end

    echo ""
    echo "######################################################################"
    echo "✓ Alpha ${STEERING_ALPHA} 的所有测试完成！"
    echo "  结果保存在: ${ALPHA_RESULT_DIR}"
    echo "######################################################################"
    echo ""

done  # Alpha loop end

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "存储位置："
echo "  KV Cache: ${CACHE_DIR}"
echo "  Results Base: ${RESULT_DIR}"
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ] && [ ! -z "${KV_STATS_PATH}" ]; then
    echo "  Steering Vectors: ${KV_STATS_PATH}"
fi
echo ""
echo "测试的参数："
echo "  Alpha 列表: ${STEERING_ALPHA_LIST[@]}"
echo "  Rate 列表: ${RATE_LIST[@]}"
echo "  总测试数: $((${#STEERING_ALPHA_LIST[@]} * ${#RATE_LIST[@]}))"
echo ""
echo "结果目录："
for ALPHA in "${STEERING_ALPHA_LIST[@]}"; do
    echo "  Alpha ${ALPHA}: ${RESULT_DIR}/alpha_${ALPHA}/"
done
echo ""
