#!/bin/bash

#####################################################################
# FusionRAG 测试启动脚本
#
# 使用方法：
#   1. 在下面的"实验配置"部分修改参数
#   2. 运行: ./run_fusionrag.sh
#####################################################################

#####################################################################
# 实验配置 - 在这里修改你的参数
#####################################################################

# GPU 配置
GPUS="0"                                                    # 使用的GPU编号，多GPU用逗号分隔如 "0,1"
USE_MULTI_GPU="false"                                       # 是否使用多GPU模式 (true/false)
cd /home/shm/document/exp/FusionRAG
# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py"
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/"       # 结果保存目录（CSV等）
CACHE_PATH="/mnt/data3/tmp/fusionrag"                      # KV cache 保存路径

# 模型配置
MODEL_TYPE="qwen"                                          # 模型类型: qwen, qwen2, qwen3, mistral, llama, pangu
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"          # 主模型路径
MODEL_NAME="Qwen2.5-7B-Instruct"                           # 模型名称（用于结果目录命名）
DRAFT_MODEL_PATH=""                                        
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"             

# 数据配置
DATA_PATH="./data/result_reflect.json"                     # 数据集路径
DATASET_NAME="musique"                                     # 数据集名称
MAX_SAMPLES=""                                             # 最大测试样本数（留空=全部）

# FusionRAG 方法配置
REPROCESS_METHOD="FusionRAG"                               # 方法: FusionRAG, Oracle, OracleAdaptive, OracleDynamic, vAttention, DraftModel, QueryAttention
RATE="0.15"                                                 # 重算比例 (0-1)
TOPK="10"                                                  # 预处理时融合的top-k文档数
PREPROCESS="true"                                          # 是否启用预处理 (true/false)
USE_RANDOM_RECALL="false"                                  # 是否使用随机召回 (true=随机, false=BGE相似度)
RANDOM_SEED="42"                                           # 随机种子 (当 USE_RANDOM_RECALL=true 时生效)
REVERT_ROPE="true"                                         # 是否还原RoPE (true/false)
PREPROCESS_SCOPE="global"                                  # 预处理范围: global, per_example, skip_untested

# 特殊方法参数
USE_ENTROPY_SELECTION="false"                              # 是否使用熵选层 (true/false)
ENTROPY_TOP_K="4"                                          # 熵选层数量
DRAFT_LAYER_SELECTION="entropy"                            # Draft模型选层方式: entropy, last, fixed, middle

# vAttention 参数
VATTENTION_TOPK_RATIO="0.5"                                # vAttention top-k比例

# OracleDynamic 参数
EPSILON="0.1"                                              # 误差容忍度
DELTA="0.05"                                               # 置信度
MIN_RATE="0.05"                                            # 最小重算比例
MAX_RATE="0.5"                                             # 最大重算比例

# Long Decode 参数
LONG_DECODE="false"                                        # 是否启用长输出模式 (true/false)
LONG_DECODE_MAX_TOKENS="1000"                              # 长输出模式最大tokens

# OpenAI 评判配置
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_API_KEY="sk-519d391217894b6e91e7c2ebf2a9f4df"
OPENAI_MODEL="deepseek-chat"

#####################################################################
# 以下是脚本逻辑，通常不需要修改
#####################################################################

# 显示帮助信息
show_help() {
    cat << EOF
FusionRAG 测试启动脚本

使用方法:
    1. 编辑脚本开头的"实验配置"部分
    2. 运行: $0

或者使用命令行参数覆盖配置:
    $0 [--gpus GPUS] [PYTHON_ARGS...]

示例:
    # 使用脚本中的配置运行
    $0

    # 覆盖 GPU 配置
    $0 --gpus 0

    # 覆盖多个参数
    $0 --gpus 0,1 --rate 0.5 --reprocess_method Oracle

    # 查看 Python 脚本的完整参数列表
    $0 --help-script
EOF
}

# 解析命令行参数（可选，用于覆盖配置）
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --help-script)
            echo "Python 脚本完整参数列表:"
            echo "========================================"
            ${PYTHON_PATH} ${SCRIPT_PATH} --help
            exit 0
            ;;
        *)
            # 将其他所有参数传递给 Python 脚本
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# 如果没有命令行参数，使用配置文件中的参数
if [ ${#PYTHON_ARGS[@]} -eq 0 ]; then
    PYTHON_ARGS=(
        "--model_type" "${MODEL_TYPE}"
        "--model_path" "${MODEL_PATH}"
        "--model_name" "${MODEL_NAME}"
        "--data_path" "${DATA_PATH}"
        "--dataset_name" "${DATASET_NAME}"
        "--cache_path" "${CACHE_PATH}"
        "--result_path" "${RESULT_DIR}"
        "--bge_model_path" "${BGE_MODEL_PATH}"
        "--rate" "${RATE}"
        "--topk" "${TOPK}"
        "--preprocess" "${PREPROCESS}"
        "--use_random_recall" "${USE_RANDOM_RECALL}"
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
        "--epsilon" "${EPSILON}"
        "--delta" "${DELTA}"
        "--min_rate" "${MIN_RATE}"
        "--max_rate" "${MAX_RATE}"
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
fi

# 创建必要的目录
mkdir -p "${RESULT_DIR}"
mkdir -p "${CACHE_PATH}"

# 打印配置信息
echo "=========================================="
echo "FusionRAG 测试"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${REPROCESS_METHOD}"
echo "Rate: ${RATE}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_PATH}"
echo "Results: ${RESULT_DIR}"
echo "=========================================="
echo ""

# 设置 GPU 并运行
export CUDA_VISIBLE_DEVICES=${GPUS}

# 运行脚本（不保存日志，直接输出到控制台）
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "运行完成！"
    echo "结果保存在: ${CACHE_PATH}"
else
    echo "运行失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
