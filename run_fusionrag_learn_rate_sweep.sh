#!/bin/bash


# 基础配置
GPUS="7"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
CACHE_DIR="/mnt/data3/tmp/fusionrag"
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 数据配置 - 同数据集实验
DATASET="musique"                                    # 数据集名称
TEST_DATA_PATH="./data/result_reflect.json"         # 完整测试数据路径
BASE_RESULT_DIR="/home/shm/document/exp/FusionRAG/result/steering_same_dataset_learn_rate_sweep"  # 结果基础目录
STEERING_OUTPUT_DIR="/mnt/data3/tmp/fusionrag/kv_stats"  # Steering vectors 输出目录

# FusionRAG 方法配置
REPROCESS_METHOD="FusionRAG"
TOPK="10"
PREPROCESS="true"
RECALL_METHOD="no_preprocess_with_bias"
RANDOM_SEED="42"

# Steering Vector 配置
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
PER_HEAD_STATS="false"  # 是否使用per-head统计（建议先用false快速测试）

# 参数扫描配置
LEARN_RATE_LIST=(10 20 30 50 80)  # 学习样本比例列表（百分比），100表示使用所有样本
STEERING_ALPHA_LIST=(0.3 0.5 0.7 1.0)  # Steering强度列表
RATE_LIST=(0.0)  # KV cache compression rate

# 层选择配置
STEERING_KEY_LAYERS="all"
STEERING_VALUE_LAYERS="all"

# 其他配置
REVERT_ROPE="true"
PREPROCESS_SCOPE="global"
USE_MULTI_GPU="false"
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

# 可选参数
MAX_SAMPLES=""
DRAFT_MODEL_PATH=""

#####################################################################
# 辅助函数
#####################################################################

# 函数：获取数据集的可用样本数量
get_available_samples() {
    local dataset=$1
    local model_name=$2
    local cache_dir=$3

    # 检查no_preprocess KV cache目录
    local kv_cache_path="${cache_dir}/${model_name}/${dataset}/kv_cache"

    if [ ! -d "${kv_cache_path}" ]; then
        echo "0"
        return
    fi

    # 统计chunk_id=1的key文件数量
    local count=$(ls "${kv_cache_path}"/*_1_key.pt 2>/dev/null | wc -l)
    echo "${count}"
}

# 函数：根据learn_rate计算需要的样本数量
calculate_sample_count() {
    local total_samples=$1
    local learn_rate=$2  # 百分比

    # 计算样本数量 (向上取整)
    local sample_count=$(( (total_samples * learn_rate + 99) / 100 ))

    # 至少使用1个样本
    if [ ${sample_count} -lt 1 ]; then
        sample_count=1
    fi

    # 不能超过总样本数
    if [ ${sample_count} -gt ${total_samples} ]; then
        sample_count=${total_samples}
    fi

    echo "${sample_count}"
}

# 函数：生成样本ID列表（0到n-1）
generate_sample_ids() {
    local count=$1
    if [ ${count} -le 0 ]; then
        echo ""
        return
    fi
    # 使用seq命令高效生成序列（从0到count-1）
    seq -s ' ' 0 $((count - 1))
}

#####################################################################
# 开始实验
#####################################################################

echo ""
echo "######################################################################"
echo "#                 FusionRAG Learn Rate Sweep 实验"
echo "######################################################################"
echo ""
echo "实验目标: 研究用少量样本学习的steering vectors的泛化能力"
echo ""
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "数据集: ${DATASET}"
echo "方法: ${REPROCESS_METHOD}"
echo "召回方法: ${RECALL_METHOD}"
echo ""
echo "实验配置:"
echo "  Learn Rate 列表: ${LEARN_RATE_LIST[@]}% (用于计算steering vectors的样本比例)"
echo "  Alpha 列表: ${STEERING_ALPHA_LIST[@]} (steering强度)"
echo "  Rate 列表: ${RATE_LIST[@]} (KV cache压缩率)"
echo "  PCA 阈值: ${PCA_VARIANCE_THRESHOLD}"
echo "  Per-head 统计: ${PER_HEAD_STATS}"
echo ""

# 获取数据集的总样本数
echo "=========================================="
echo "检查数据集可用样本..."
echo "=========================================="
TOTAL_SAMPLES=$(get_available_samples "${DATASET}" "${MODEL_NAME}" "${CACHE_DIR}")

if [ ${TOTAL_SAMPLES} -eq 0 ]; then
    echo "✗ 错误: 未找到 ${DATASET} 的KV cache文件"
    echo "  路径: ${CACHE_DIR}/${MODEL_NAME}/${DATASET}/kv_cache/"
    echo "  请先运行数据预处理生成KV cache"
    exit 1
fi

echo "✓ 数据集 ${DATASET} 可用样本数: ${TOTAL_SAMPLES}"
echo ""

# 计算总测试数
TOTAL_TESTS=$((${#LEARN_RATE_LIST[@]} * ${#STEERING_ALPHA_LIST[@]} * ${#RATE_LIST[@]}))
echo "将进行 ${TOTAL_TESTS} 次测试:"
echo "  ${#LEARN_RATE_LIST[@]} 个 Learn Rates × ${#STEERING_ALPHA_LIST[@]} 个 Alphas × ${#RATE_LIST[@]} 个 Rates"
echo ""

# 创建必要的目录
mkdir -p "${BASE_RESULT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${STEERING_OUTPUT_DIR}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

#####################################################################
# 遍历 Learn Rate
#####################################################################

for LEARN_RATE in "${LEARN_RATE_LIST[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Learn Rate = ${LEARN_RATE}% (用${LEARN_RATE}%样本学习，在100%样本上测试)"
    echo "######################################################################"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 计算该learn_rate需要的样本数量
    SAMPLE_COUNT=$(calculate_sample_count ${TOTAL_SAMPLES} ${LEARN_RATE})
    echo "样本配置:"
    echo "  总样本数: ${TOTAL_SAMPLES}"
    echo "  Learn Rate: ${LEARN_RATE}%"
    echo "  学习样本数: ${SAMPLE_COUNT} (前${SAMPLE_COUNT}个样本用于计算steering vectors)"
    echo "  测试样本数: ${TOTAL_SAMPLES} (在完整数据集上测试)"
    echo ""

    # 为这个learn_rate创建结果目录
    LEARN_RATE_RESULT_DIR="${BASE_RESULT_DIR}/learn_rate_${LEARN_RATE}pct"
    mkdir -p "${LEARN_RATE_RESULT_DIR}"

    #####################################################################
    # Step 0: 计算 Steering Vectors (使用 LEARN_RATE% 的样本)
    #####################################################################

    echo "=========================================="
    echo "Step 0: 计算 Steering Vectors"
    echo "=========================================="
    echo "数据集: ${DATASET}"
    echo "模型: ${MODEL_NAME}"
    echo "学习样本数: ${SAMPLE_COUNT} / ${TOTAL_SAMPLES} (${LEARN_RATE}%)"
    echo ""

    # 生成文件名（包含learn_rate和sample_count信息）
    if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
        SUFFIX="manifold"
        PCA_SUFFIX=$(echo "${PCA_VARIANCE_THRESHOLD}" | sed 's/0\.//' | sed 's/\.//')
        SUFFIX="${SUFFIX}_pca${PCA_SUFFIX}"
    else
        SUFFIX="no_proj"
    fi

    if [ "${PER_HEAD_STATS}" = "true" ]; then
        SUFFIX="${SUFFIX}_perhead"
    fi

    # 文件名包含：数据集_学习样本数_总样本数_learn_rate
    KV_STATS_PATH="${STEERING_OUTPUT_DIR}/${DATASET}_learn${SAMPLE_COUNT}of${TOTAL_SAMPLES}_${LEARN_RATE}pct_${SUFFIX}.pt"

    echo "输出路径: ${KV_STATS_PATH}"
    echo ""

    # 检查文件是否已存在
    if [ -f "${KV_STATS_PATH}" ]; then
        echo "✓ Steering vectors 文件已存在，跳过计算"
        echo "  ${KV_STATS_PATH}"
        echo "  如需重新计算，请删除此文件"
    else
        echo "开始计算 steering vectors..."
        echo ""

        # 生成样本ID列表 (0到SAMPLE_COUNT-1)
        STEERING_SAMPLE_IDS=$(generate_sample_ids ${SAMPLE_COUNT})

        # 构建命令
        CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
            --cache_dir \"${CACHE_DIR}\" \
            --dataset \"${DATASET}\" \
            --model_name \"${MODEL_NAME}\" \
            --sample_ids ${STEERING_SAMPLE_IDS} \
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
            echo "  Per-head statistics: 启用"
        else
            echo "  Per-head statistics: 禁用"
        fi

        echo ""
        echo "使用样本ID: ${STEERING_SAMPLE_IDS}"
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

    #####################################################################
    # 遍历 Alpha 和 Rate (在完整数据集上测试)
    #####################################################################

    for STEERING_ALPHA in "${STEERING_ALPHA_LIST[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "测试: Learn Rate=${LEARN_RATE}%, Alpha=${STEERING_ALPHA}"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"
        echo ""

        # 为每个alpha创建独立目录
        ALPHA_RESULT_DIR="${LEARN_RATE_RESULT_DIR}/alpha_${STEERING_ALPHA}"
        mkdir -p "${ALPHA_RESULT_DIR}"

        for RATE in "${RATE_LIST[@]}"; do
            echo "开始测试 Rate = ${RATE}..."

            # 构建参数列表
            PYTHON_ARGS=(
                "--model_type" "${MODEL_TYPE}"
                "--model_path" "${MODEL_PATH}"
                "--model_name" "${MODEL_NAME}"
                "--data_path" "${TEST_DATA_PATH}"
                "--dataset_name" "${DATASET}"
                "--cache_path" "${CACHE_DIR}"
                "--result_path" "${ALPHA_RESULT_DIR}"
                "--bge_model_path" "${BGE_MODEL_PATH}"
                "--rate" "${RATE}"
                "--topk" "${TOPK}"
                "--preprocess" "${PREPROCESS}"
                "--recall_method" "${RECALL_METHOD}"
                "--random_seed" "${RANDOM_SEED}"
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

            # Steering vector 相关参数
            PYTHON_ARGS+=("--kv_stats_path" "${KV_STATS_PATH}")
            PYTHON_ARGS+=("--steering_alpha" "${STEERING_ALPHA}")
            PYTHON_ARGS+=("--steering_key_layers" "${STEERING_KEY_LAYERS}")
            PYTHON_ARGS+=("--steering_value_layers" "${STEERING_VALUE_LAYERS}")

            if [ "${PER_HEAD_STATS}" = "true" ]; then
                PYTHON_ARGS+=("--use_per_head_steering")
            fi

            # 运行测试
            ${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

            EXIT_CODE=$?

            if [ ${EXIT_CODE} -eq 0 ]; then
                echo "✓ Learn Rate ${LEARN_RATE}%, Alpha ${STEERING_ALPHA}, Rate ${RATE} 测试完成"
            else
                echo "✗ Learn Rate ${LEARN_RATE}%, Alpha ${STEERING_ALPHA}, Rate ${RATE} 测试失败！退出码: ${EXIT_CODE}"
            fi

            echo ""
        done  # Rate loop
    done  # Alpha loop

    echo ""
    echo "######################################################################"
    echo "✓ Learn Rate ${LEARN_RATE}% 的所有测试完成！"
    echo "  结果保存在: ${LEARN_RATE_RESULT_DIR}/"
    echo "######################################################################"
    echo ""

done  # Learn Rate loop

#####################################################################
# 最终总结
#####################################################################

echo ""
echo "========================================================================"
echo "                       所有测试完成！"
echo "========================================================================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "实验总结:"
echo "  数据集: ${DATASET}"
echo "  总样本数: ${TOTAL_SAMPLES}"
echo "  Learn Rate 列表: ${LEARN_RATE_LIST[@]}%"
echo "  Alpha 列表: ${STEERING_ALPHA_LIST[@]}"
echo "  Rate 列表: ${RATE_LIST[@]}"
echo "  总测试数: ${TOTAL_TESTS}"
echo ""
echo "存储位置:"
echo "  KV Cache: ${CACHE_DIR}/${MODEL_NAME}/${DATASET}/"
echo "  Steering Vectors: ${STEERING_OUTPUT_DIR}/"
echo "  Results: ${BASE_RESULT_DIR}/"
echo ""
echo "结果目录结构:"
for LEARN_RATE in "${LEARN_RATE_LIST[@]}"; do
    echo "  Learn Rate ${LEARN_RATE}%: ${BASE_RESULT_DIR}/learn_rate_${LEARN_RATE}pct/"
    for ALPHA in "${STEERING_ALPHA_LIST[@]}"; do
        echo "    ├─ Alpha ${ALPHA}: learn_rate_${LEARN_RATE}pct/alpha_${ALPHA}/"
    done
done
echo ""
echo "Steering Vectors 文件:"
for LEARN_RATE in "${LEARN_RATE_LIST[@]}"; do
    SAMPLE_COUNT=$(calculate_sample_count ${TOTAL_SAMPLES} ${LEARN_RATE})
    echo "  Learn Rate ${LEARN_RATE}% (${SAMPLE_COUNT} samples):"

    if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
        SUFFIX="manifold"
        PCA_SUFFIX=$(echo "${PCA_VARIANCE_THRESHOLD}" | sed 's/0\.//' | sed 's/\.//')
        SUFFIX="${SUFFIX}_pca${PCA_SUFFIX}"
    else
        SUFFIX="no_proj"
    fi

    if [ "${PER_HEAD_STATS}" = "true" ]; then
        SUFFIX="${SUFFIX}_perhead"
    fi

    FILE="${STEERING_OUTPUT_DIR}/${DATASET}_learn${SAMPLE_COUNT}of${TOTAL_SAMPLES}_${LEARN_RATE}pct_${SUFFIX}.pt"
    echo "    ${FILE}"
done
echo ""
echo "========================================================================"
echo ""
