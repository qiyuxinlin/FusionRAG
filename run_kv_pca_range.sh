#!/bin/bash

#####################################################################
# KV Cache PCA 范围分析脚本
#
# 使用方法：
#   bash run_kv_pca_range.sh <start_id> <end_id> [output_dir]
#
# 示例：
#   bash run_kv_pca_range.sh 0 5           # 分析样本 0-5
#   bash run_kv_pca_range.sh 10 20         # 分析样本 10-20
#   bash run_kv_pca_range.sh 0 9 ./out    # 分析样本 0-9，保存到 ./out
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 解析参数
START_ID=${1:-0}
END_ID=${2:-5}
OUTPUT_DIR=${3:-"./kv_pca_analysis"}

# 生成样本 ID 列表
SAMPLE_IDS=$(seq ${START_ID} ${END_ID})
NUM_SAMPLES=$((END_ID - START_ID + 1))

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_pca.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 分析配置
CHUNK_ID="1"                 # Chunk ID (1=第一个文档, 0=system prompt)
MAX_LAYERS="28"              # 模型总层数
MAX_TOKENS="500"             # 每层采样的最大token数
LAYERS="0 5 11 16 22 27"     # 分析的层

echo "=========================================="
echo "KV Cache PCA 范围分析"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "样本范围: ${START_ID} - ${END_ID} (共 ${NUM_SAMPLES} 个)"
echo "分析层: ${LAYERS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行分析
${PYTHON_PATH} ${SCRIPT_PATH} \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --sample_ids ${SAMPLE_IDS} \
    --chunk_id ${CHUNK_ID} \
    --layers ${LAYERS} \
    --max_layers ${MAX_LAYERS} \
    --max_tokens ${MAX_TOKENS} \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 分析完成！"
    echo ""
    echo "分析了 ${NUM_SAMPLES} 个样本"
    echo ""
    echo "生成的可视化文件："
    echo "  - 每个样本的 PCA 散点图"
    echo "  - l2_distance_trends.png (趋势图)"
    echo "  - pca_variance_explained.png (方差图)"
    echo "  - summary_statistics.json (统计)"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
