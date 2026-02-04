#!/bin/bash


# 基础配置
GPUS="4"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v3.py"  # 修改为 v3 版本
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/self_supervised"   # 自监督模式结果目录
CACHE_DIR="/mnt/data3/tmp/fusionrag_online_lazy"               # KV cache 保存路径
cd /home/shm/document/exp/FusionRAG

# 模型配置
MODEL_TYPE="qwen"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 数据集配置
DATASET_NAME="musique-susvised"  # 数据集名称
# DATA_PATH="./data/result_musique_reflect_optimized.json" 
# DATASET_NAME="musique" 
# 自监督模式核心配置
RECALL_METHOD="self_supervised"  # 关键：使用自监督模式
REVERT_ROPE="true"               # RoPE调整

# 可选参数
MAX_SAMPLES="1"  # 留空表示测试所有文档，或设置为数字（如"100"）


echo "=========================================="
echo "Self-Supervised Repeat Task"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${GPUS}"
echo "模型: ${MODEL_NAME}"
echo "召回方法: ${RECALL_METHOD}"
echo "数据集: ${DATASET_NAME}"
echo "KV Cache: ${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache"
echo "Results: ${RESULT_DIR}"
if [ ! -z "${MAX_SAMPLES}" ]; then
    echo "最大样本数: ${MAX_SAMPLES}"
fi
echo "=========================================="
echo ""

# 创建必要的目录
mkdir -p "${RESULT_DIR}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${GPUS}

# 记录开始时间
START_TIME=$(date +%s)

echo "开始测试..."
echo ""

# 构建参数列表
PYTHON_ARGS=(
    "--model_type" "${MODEL_TYPE}"
    "--model_path" "${MODEL_PATH}"
    "--model_name" "${MODEL_NAME}"
    "--dataset_name" "${DATASET_NAME}"
    "--cache_path" "${CACHE_DIR}"
    "--result_path" "${RESULT_DIR}"
    "--recall_method" "${RECALL_METHOD}"
    "--revert_rope" "${REVERT_ROPE}"
    "--device" "cuda:0"
)

# 添加可选参数
if [ ! -z "${MAX_SAMPLES}" ]; then
    PYTHON_ARGS+=("--max_samples" "${MAX_SAMPLES}")
fi

# 运行测试
${PYTHON_PATH} ${SCRIPT_PATH} "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "=========================================="
    echo "测试完成！"
    echo "=========================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "总耗时: ${MINUTES}分${SECONDS}秒"
    echo "=========================================="
    echo ""
    echo "结果保存在: ${RESULT_DIR}/self_supervised_results.json"
    echo ""

    # 显示 KV cache 统计
    if [ -d "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" ]; then
        CACHE_COUNT=$(find "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" -name "*_key.pt" 2>/dev/null | wc -l)
        if [ ${CACHE_COUNT} -gt 0 ]; then
            CACHE_SIZE=$(du -sh "${CACHE_DIR}/${MODEL_NAME}/${DATASET_NAME}/kv_cache" 2>/dev/null | cut -f1)
            echo "KV Cache 统计："
            echo "  文档数: ${CACHE_COUNT}"
            echo "  大小: ${CACHE_SIZE}"
            echo ""
        fi
    fi
else
    echo "=========================================="
    echo "测试失败！退出码: ${EXIT_CODE}"
    echo "=========================================="
fi

echo ""
