#!/bin/bash

#####################################################################
# KV Cache PCA Visualization Script
#
# 分析 No Preprocess 和 BGE 两种模式下 KV cache 的特征分布
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_pca.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"
OUTPUT_DIR="./kv_pca_analysis3"

# 分析配置（可通过命令行参数覆盖）
SAMPLE_IDS="${1:-16}"  # 要分析的样本ID
CHUNK_ID="1"                 # Chunk ID (1=第一个文档, 0=system prompt)
MAX_LAYERS="28"              # 模型总层数
MAX_TOKENS="500"             # 每层采样的最大token数

# 选择要分析的层（可选）
# 如果不指定，会自动选择均匀分布的6层
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27" 

echo "=========================================="
echo "KV Cache PCA 分析"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "样本 IDs: ${SAMPLE_IDS}"
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
    echo "生成的可视化文件："
    echo "  1. pca_key_example*_chunk*.png  - Key cache PCA 分布图"
    echo "  2. pca_value_example*_chunk*.png - Value cache PCA 分布图"
    echo "  3. l2_distance_trends.png - 各层L2距离趋势"
    echo "  4. pca_variance_explained.png - PCA方差解释比例"
    echo "  5. summary_statistics.json - 统计摘要"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
