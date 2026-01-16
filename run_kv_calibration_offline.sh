#!/bin/bash

#####################################################################
# KV Calibration - Offline Mode
# 统计 BGE preprocess 和 no_preprocess 之间的KV偏移分布
#####################################################################

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/kv_calibration.py"
cd /home/shm/document/exp/FusionRAG

# 配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"
NUM_LAYERS=28

# Offline统计参数
SAMPLE_RATIO="0.1"           # 使用10%的样本统计偏移
REFERENCE_METHOD="bge"        # 参考方法（preprocess方法）
GRANULARITY="per_layer"       # 统计粒度：per_layer | per_head | per_position
AGGREGATION="mean"            # 聚合方式
AUTO_SELECT_LAYERS="false"    # 是否自动选择显著层
THRESHOLD="0.1"               # 自动选择阈值

# 输出路径（自动生成）
STATS_PATH="${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_${REFERENCE_METHOD}_${GRANULARITY}.pt"

echo "=========================================="
echo "KV Calibration - Offline Mode"
echo "=========================================="
echo "Cache Dir: ${CACHE_DIR}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Reference Method: ${REFERENCE_METHOD}"
echo "Granularity: ${GRANULARITY}"
echo "Sample Ratio: ${SAMPLE_RATIO}"
echo "Auto Select Layers: ${AUTO_SELECT_LAYERS}"
echo "Stats Path: ${STATS_PATH}"
echo "=========================================="
echo ""

# 运行offline统计
${PYTHON_PATH} ${SCRIPT_PATH} \
    --mode offline \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --num_layers ${NUM_LAYERS} \
    --sample_ratio ${SAMPLE_RATIO} \
    --reference_method "${REFERENCE_METHOD}" \
    --granularity "${GRANULARITY}" \
    --aggregation "${AGGREGATION}" \
    --auto_select_layers "${AUTO_SELECT_LAYERS}" \
    --threshold ${THRESHOLD} \
    --stats_path "${STATS_PATH}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Offline统计完成！"
    echo "=========================================="
    echo "统计量已保存到:"
    echo "  ${STATS_PATH}"
    echo "  ${STATS_PATH%.pt}.json"
    echo ""
    echo "下一步："
    echo "1. 查看统计摘要:"
    echo "   ${PYTHON_PATH} ${SCRIPT_PATH} --mode summary --stats_path ${STATS_PATH}"
    echo ""
    echo "2. 使用Online模式进行推理（修改 run_fusionrag_sweep.sh）:"
    echo "   ENABLE_KV_CALIBRATION=\"true\""
    echo "   KV_CALIBRATION_MODE=\"online\""
    echo "   CALIBRATION_STATS_PATH=\"${STATS_PATH}\""
    echo "=========================================="
else
    echo ""
    echo "✗ Offline统计失败！退出码: ${EXIT_CODE}"
fi
