#!/bin/bash

#####################################################################
# KV Calibration 完整演示流程
# 展示从offline统计到online应用的完整workflow
#####################################################################

set -e  # 遇到错误立即退出

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
cd /home/shm/document/exp/FusionRAG

CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

echo "=========================================="
echo "KV Calibration 完整演示流程"
echo "=========================================="
echo ""
echo "本脚本将演示："
echo "1. Offline阶段：统计BGE和no_preprocess之间的KV偏移"
echo "2. 查看统计摘要"
echo "3. 如何配置Online阶段"
echo ""
echo "按 Enter 继续..."
read

# ========================================
# Step 1: Offline统计（三种粒度）
# ========================================
echo ""
echo "=========================================="
echo "Step 1: Offline统计 - 不同粒度对比"
echo "=========================================="
echo ""

# 1.1 per_layer粒度
echo "1.1 per_layer粒度统计 (推荐开始)..."
${PYTHON_PATH} kv_calibration.py \
    --mode offline \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --num_layers 28 \
    --sample_ratio 0.1 \
    --reference_method "bge" \
    --granularity "per_layer" \
    --aggregation "mean" \
    --auto_select_layers "false" \
    --threshold 0.1 \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer.pt"

echo ""
echo "✓ per_layer统计完成"
echo ""

# 1.2 per_head粒度
echo "1.2 per_head粒度统计 (更精细)..."
${PYTHON_PATH} kv_calibration.py \
    --mode offline \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --num_layers 28 \
    --sample_ratio 0.1 \
    --reference_method "bge" \
    --granularity "per_head" \
    --aggregation "mean" \
    --auto_select_layers "false" \
    --threshold 0.1 \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_head.pt"

echo ""
echo "✓ per_head统计完成"
echo ""

# 1.3 自适应层选择
echo "1.3 per_layer + 自适应层选择..."
${PYTHON_PATH} kv_calibration.py \
    --mode offline \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --num_layers 28 \
    --sample_ratio 0.1 \
    --reference_method "bge" \
    --granularity "per_layer" \
    --aggregation "mean" \
    --auto_select_layers "true" \
    --threshold 0.1 \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer_auto.pt"

echo ""
echo "✓ 自适应层选择统计完成"
echo ""

# ========================================
# Step 2: 查看统计摘要
# ========================================
echo ""
echo "=========================================="
echo "Step 2: 查看统计摘要"
echo "=========================================="
echo ""

echo "2.1 per_layer摘要:"
echo "-------------------"
${PYTHON_PATH} kv_calibration.py \
    --mode summary \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer.pt"

echo ""
echo "2.2 per_head摘要:"
echo "-------------------"
${PYTHON_PATH} kv_calibration.py \
    --mode summary \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_head.pt"

echo ""
echo "2.3 自适应层选择摘要:"
echo "-------------------"
${PYTHON_PATH} kv_calibration.py \
    --mode summary \
    --stats_path "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer_auto.pt"

# ========================================
# Step 3: Online配置说明
# ========================================
echo ""
echo "=========================================="
echo "Step 3: Online阶段配置说明"
echo "=========================================="
echo ""
echo "现在你可以在 run_fusionrag_sweep.sh 中配置："
echo ""
echo "# 启用KV校准"
echo "ENABLE_KV_CALIBRATION=\"true\""
echo "KV_CALIBRATION_MODE=\"online\""
echo ""
echo "# 选择一个统计量文件（根据Step 2的结果）"
echo "# 方案1: per_layer (推荐)"
echo "CALIBRATION_STATS_PATH=\"${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer.pt\""
echo ""
echo "# 方案2: per_head (更精细)"
echo "# CALIBRATION_STATS_PATH=\"${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_head.pt\""
echo ""
echo "# 方案3: 自适应层选择"
echo "# CALIBRATION_STATS_PATH=\"${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_bge_per_layer_auto.pt\""
echo ""
echo "# (可选) 手动指定要校准的层"
echo "# CALIBRATION_KEY_LAYERS=\"0,1,2,3,4,5,6,7,8,9\""
echo "# CALIBRATION_VALUE_LAYERS=\"0,1,2,3,4,5,6,7,8,9\""
echo ""
echo "然后运行："
echo "bash run_fusionrag_sweep.sh"
echo ""

# ========================================
# 总结
# ========================================
echo "=========================================="
echo "演示完成！"
echo "=========================================="
echo ""
echo "生成的文件："
ls -lh "${CACHE_DIR}/${DATASET}/${MODEL_NAME}/calibration_stats_"*
echo ""
echo "详细使用说明请查看："
echo "  KV_CALIBRATION_GUIDE.md"
echo ""
echo "=========================================="
