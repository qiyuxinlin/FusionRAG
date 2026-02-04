#!/bin/bash
# KV Cache 重建实验启动脚本
# Author: Claude Code
# Version: 1.0.0

# ============================================================
# 配置参数
# ============================================================

# Python 环境
PYTHON="/home/shm/anaconda3/envs/fusionrag/bin/python"

# 模型路径
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"

# 数据路径
DATA_DIR="/home/shm/document/exp/FusionRAG"
DATA_PATH="${DATA_DIR}/data/2wiki_input_rebuilt.json"
CACHE_PATH="${DATA_DIR}/cache/rebuilt_exp/"
OUTPUT_DIR="${DATA_DIR}/results"

# BGE 模型用于语义相似度计算
BGE_MODEL_PATH="/mnt/data/models/bge-m3-FP16"

# 实验参数
K=5                    # 目标文档前的随机文档数量
NUM_EXPERIMENTS=10     # 实验次数
MAX_NEW_TOKENS=512     # 最大生成 token 数
DEVICE="cuda:0"        # 设备

# 阈值
F1_THRESHOLD=0.9               # 字面匹配的 F1 分数阈值
SEMANTIC_THRESHOLD=0.85        # 语义相似度阈值

# Prompt
# 使用 JSON 格式输出，简化内容提取
REPEAT_PROMPT="
Instruction: Repeat the above text word-for-word. Output your response as JSON with key 'repeated_text'."

# ============================================================
# 解析命令行参数
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -K|--K)
            K="$2"
            shift 2
            ;;
        -n|--num_experiments)
            NUM_EXPERIMENTS="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --multi_gpu)
            USE_MULTI_GPU="--use_multi_gpu"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -K, --K 值                     目标文档前的随机文档数量 (默认: 5)"
            echo "  -n, --num_experiments 数值      实验次数 (默认: 10)"
            echo "  --model_path 路径               模型路径"
            echo "  --device 设备                   使用的设备 (默认: cuda:0)"
            echo "  --output_dir 路径               输出目录 (默认: ./results/)"
            echo "  --multi_gpu                     使用多 GPU"
            echo "  -h, --help                      显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 运行 K=5, 10 次实验"
            echo "  $0 -K 5 -n 10"
            echo ""
            echo "  # 使用多 GPU 运行"
            echo "  $0 -K 5 -n 20 --multi_gpu"
            echo ""
            echo "  # 批量测试多个 K 值"
            echo "  $0 --batch 0,1,2,5,10"
            exit 0
            ;;
        --batch)
            # 批量模式：测试多个 K 值
            K_VALUES="$2"
            BATCH_MODE=true
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看用法信息"
            exit 1
            ;;
    esac
done

# ============================================================
# 函数
# ============================================================

run_experiment() {
    local k=$1
    local num_exp=$2
    local output_file="${OUTPUT_DIR}/rebuilt_k${k}.json"

    echo "========================================"
    echo "运行 KV Cache 重建实验"
    echo "========================================"
    echo "K (随机文档数):           ${k}"
    echo "实验次数:                 ${num_exp}"
    echo "模型:                     ${MODEL_PATH}"
    echo "设备:                     ${DEVICE}"
    echo "输出文件:                 ${output_file}"
    echo "========================================"
    echo ""

    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"

    # 运行实验
    ${PYTHON} script/rebulit_exp/kv_cache_rebuilt_exp.py \
        --K ${k} \
        --num_experiments ${num_exp} \
        --model_type ${MODEL_TYPE} \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --cache_path "${CACHE_PATH}" \
        --device "${DEVICE}" \
        ${USE_MULTI_GPU} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --repeat_prompt "${REPEAT_PROMPT}" \
        --output_file "${output_file}" \
        --revert_rope \
        --bge_model_path "${BGE_MODEL_PATH}" \
        --f1_threshold ${F1_THRESHOLD} \
        --semantic_threshold ${SEMANTIC_THRESHOLD}

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "实验成功完成！"
        echo "结果已保存到: ${output_file}"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "实验失败，退出码: ${exit_code}"
        echo "========================================"
    fi

    return ${exit_code}
}

# ============================================================
# 主程序
# ============================================================

cd "${DATA_DIR}" || exit 1

if [ "${BATCH_MODE}" = true ]; then
    # 批量模式：为多个 K 值运行实验
    echo "========================================"
    echo "批量模式: 测试 K = ${K_VALUES}"
    echo "========================================"
    echo ""

    # 将逗号分隔的值转换为数组
    IFS=',' read -ra K_ARRAY <<< "${K_VALUES}"

    # 为每个 K 值运行实验
    for k in "${K_ARRAY[@]}"; do
        run_experiment ${k} ${NUM_EXPERIMENTS}
        echo ""
        echo "等待 5 秒后进行下一个实验..."
        echo ""
        sleep 5
    done

    # 打印汇总
    echo "========================================"
    echo "批量实验汇总"
    echo "========================================"
    for k in "${K_ARRAY[@]}"; do
        output_file="${OUTPUT_DIR}/rebuilt_k${k}.json"
        if [ -f "${output_file}" ]; then
            echo "K=${k}: ${output_file}"
            # 提取并显示关键指标
            if command -v jq &> /dev/null; then
                success_rate=$(jq -r '.summary.success_rate' "${output_file}")
                avg_f1=$(jq -r '.summary.avg_f1' "${output_file}")
                avg_semantic=$(jq -r '.summary.avg_semantic_similarity' "${output_file}")
                echo "  成功率: ${success_rate}"
                echo "  平均 F1:    ${avg_f1}"
                echo "  平均语义相似度: ${avg_semantic}"
            fi
        else
            echo "K=${k}: 失败 (输出文件未找到)"
        fi
        echo ""
    done
else
    # 单实验模式
    run_experiment ${K} ${NUM_EXPERIMENTS}
fi
